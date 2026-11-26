"""Mine formula-matched 2D negatives from NMRexp training spectra with an existing checkpoint."""
import argparse
import heapq
import json
import time
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds
import torch
from rdkit import rdBase

from chem import canonical, digest, negative_graph
from data import condition
from data_io import hash_id, normalize_smiles, write_json
from runtime import load_checkpoint
from search import generate, LEGACY_SEARCH


def select_rows(dataset, limit, seed, cohort):
    """Prefer paired spectra, then the lowest row ID. Hash sampling is nested across limits."""
    best = {}
    for block in dataset.to_batches(columns=['row_id', 'mol_id', 'h_peaks', 'c_peaks'],
                                    filter=ds.field('has_reference'), batch_size=8192):
        h, c = (pc.fill_null(pc.greater(pc.list_value_length(block.column(k)), 0), False)
                for k in ('h_peaks', 'c_peaks'))
        paired = pc.and_(h, c)
        keep = paired if cohort == 'paired' else pc.or_(h, c)
        selected = block.select(['mol_id', 'row_id']).append_column('paired', paired).filter(keep)
        for row in selected.to_pylist():
            mid = row['mol_id']
            key = (not row['paired'], row['row_id'])
            if mid not in best or key < best[mid]:
                best[mid] = key
    molecules = best
    if 0 < limit < len(best):
        molecules = heapq.nsmallest(limit, best, key=lambda mid: (hash_id(f'{seed}:{mid}'), mid))
    return {best[mid][1]: mid for mid in molecules}


def filter_negatives(target, formula, candidates, limit):
    target = canonical(target)
    result, seen = [], set()
    for candidate in candidates:
        try:
            info = normalize_smiles(candidate['smiles'])
        except (ValueError, TypeError, RuntimeError):
            continue
        s = info['smiles']
        if s == target or s in seen or info['formula'] != formula or negative_graph(s) is None:
            continue
        seen.add(s)
        result.append(s)
        if len(result) == limit:
            break
    return result


def previous_rows(path):
    """Keep only IDs and counts in memory. Recover an interrupted final append."""
    existing = {}
    with_candidates = total_candidates = 0
    with path.open('r+b') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                if line.endswith(b'\n'):
                    raise ValueError(f'Corrupt hard-negative record at byte {start}') from error
                stream.truncate(start)
                break
            rid = row['row_id']
            if rid in existing:
                raise ValueError(f'Duplicate cached row ID: {rid}')
            existing[rid] = (row['mol_id'], row['smiles'])
            n = len(row['negative_smiles'])
            total_candidates += n
            with_candidates += bool(n)
            if not line.endswith(b'\n'):
                stream.write(b'\n')
    return existing, with_candidates, total_candidates


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True, help='trained NMRexp checkpoint')
    p.add_argument('--data', type=Path, help='original step-03 dataset for this vocabulary')
    p.add_argument('--out', type=Path, required=True, help='hard-negative JSONL path')
    p.add_argument('--limit', type=int, default=10000, help='maximum unique molecules, 0 = all')
    p.add_argument('--max-negatives', type=int, default=8)
    p.add_argument('--beam', type=int, default=32)
    p.add_argument('--connect-beam', type=int, default=8)
    p.add_argument('--cohort', choices=('available', 'paired'),
                   help='available: H or C (default); paired: require both')
    p.add_argument('--metadata', action=argparse.BooleanOptionalAction, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if min(a.beam, a.connect_beam, a.max_negatives) < 1 or min(a.limit, a.log_every, a.threads) < 0:
        p.error('beam widths and max-negatives must be positive; limit, log-every and threads must be nonnegative')
    if a.threads:
        torch.set_num_threads(a.threads)
    rdBase.DisableLog('rdApp.warning')
    model, ck, root, manifest = load_checkpoint(a.checkpoint, a.device, a.data)
    if ck['stage'] != 'nmrexp':
        p.error('--checkpoint must be trained on NMRexp spectra')
    if manifest.get('hard_candidates') is not None:
        p.error('--data must point to the original step-03 dataset')
    model.eval()
    metadata = ck.get('train_args', {}).get('metadata', False) if a.metadata is None else a.metadata
    del ck
    config_path = a.out.with_suffix('.config.json')
    existing, with_candidates, total_candidates = {}, 0, 0
    previous = None
    if a.out.exists():
        if not a.resume:
            raise FileExistsError('Use --resume or choose a new output file')
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        previous.setdefault('cohort', 'paired')  # original miner only accepted H+C
    cohort = a.cohort or (previous['cohort'] if previous else 'available')
    mining = dict(LEGACY_SEARCH, beam=a.beam, connect_beam=a.connect_beam)
    config = dict(checkpoint_sha256=digest(a.checkpoint), data_manifest_sha256=digest(root/'manifest.json'),
                  tokenizer_sha256=digest(root/'tokenizer.json'), stage='nmrexp', split='train', mode='hc',
                  cohort=cohort, formula=True, annotations='full', metadata=metadata, seed=a.seed,
                  beam=a.beam, connect_beam=a.connect_beam, max_negatives=a.max_negatives,
                  rank_mix=mining['rank_mix'], pool_size=mining['pool_size'])
    if previous is not None:
        if previous != config:
            raise ValueError('Resume requires the same checkpoint, dataset and mining settings')
        existing, with_candidates, total_candidates = previous_rows(a.out)

    dataset = ds.dataset(str(root/'examples/source=NMRexp/split=train'), format='parquet')
    ids = select_rows(dataset, a.limit, a.seed, cohort)
    if not ids:
        raise ValueError('No eligible NMRexp training records')
    if not existing.keys() <= ids.keys():
        raise ValueError('Existing cache exceeds the selected sample; restore or increase --limit')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(config_path, config)
    n = len(existing)
    print(f'Selected {len(ids):,} training molecules; {n:,} already cached.', flush=True)
    fields = ['row_id', 'mol_id', 'smiles', 'formula', 'formula_counts', 'h_peaks', 'c_peaks',
              'h_solvent', 'c_solvent', 'h_mhz', 'c_mhz']
    start = time.monotonic()
    generated = 0
    expected = len(ids)
    with a.out.open('a', encoding='utf-8') as stream, torch.no_grad():
        # Scan only spectrum columns. Full graph arrays need not reside in memory for mining.
        query = ds.field('has_reference') & ds.field('row_id').isin(list(ids))
        for block in dataset.to_batches(columns=fields, filter=query, batch_size=512):
            for row in block.to_pylist():
                rid, mid = row['row_id'], row['mol_id']
                if ids.pop(rid, None) != mid:
                    raise ValueError(f'Duplicate or mismatched training row: {rid}')
                if rid in existing:
                    if existing[rid] != (mid, row['smiles']):
                        raise ValueError(f'Cached target changed: {rid}')
                    continue
                candidates = generate(model, condition(row, metadata=metadata),
                                      max_fragments=model.config.max_heavy, **mining)
                negatives = filter_negatives(row['smiles'], row['formula'], candidates, a.max_negatives)
                entry = dict(row_id=rid, mol_id=mid, smiles=row['smiles'], negative_smiles=negatives)
                print(json.dumps(entry, ensure_ascii=False), file=stream, flush=True)
                n += 1
                generated += 1
                with_candidates += bool(negatives)
                total_candidates += len(negatives)
                if a.log_every and generated % a.log_every == 0:
                    elapsed = time.monotonic() - start
                    print(f'n={n:,}/{expected:,} elapsed={elapsed/3600:.2f}h '
                          f'ETA={elapsed*(expected-n)/generated/3600:.2f}h', flush=True)
    if ids:
        raise ValueError(f'Missing {len(ids)} selected training rows')
    report = dict(n=n, with_negatives=with_candidates, negative_count=total_candidates,
                  generated_this_run=generated, seconds_this_run=time.monotonic()-start, config=config)
    write_json(a.out.with_suffix('.summary.json'), report)
    print(f'n={n:,} with_negatives={with_candidates:,} negatives={total_candidates:,}', flush=True)


if __name__ == '__main__':
    main()
