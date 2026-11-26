"""Evaluate ranked 2D molecular structures from one PubChem or NMRexp split."""
import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
import torch
from data import batches, eligible, SAMPLING_ID, EVAL_SEED
from data_io import write_json
from runtime import load_checkpoint
from search import (add_search_arguments, add_spectrum_arguments, search_options, resolved_options,
                    predict_record, REFERENCE_SEARCH, LEGACY_SEARCH, SPECTRUM_DEFAULTS)
from metrics import (TOMI_DESCRIPTION, TOMI_KEYS, ProgressLog, evaluate_one, format_summary,
                     mean_metrics, topk_tomi, valid_unique_smiles)
from chem import canonical, digest


def restore_metrics(path):
    """Stream saved results. Upgrade legacy rows atomically only if needed."""
    replacement = None
    try:
        with path.open('r+b') as source:
            row_count = 0
            while True:
                start = source.tell()
                line = source.readline()
                if not line:
                    break
                try:
                    record = json.loads(line)
                except (json.JSONDecodeError, UnicodeDecodeError) as error:
                    if line.endswith(b'\n'):
                        raise ValueError(f'Corrupt evaluation record at byte {start}') from error
                    if replacement is None:
                        source.truncate(start)
                    break  # interrupted final write
                previous = record['metrics']
                if any(key not in previous for key in TOMI_KEYS):
                    previous.update(topk_tomi(canonical(record['smiles']), valid_unique_smiles(record['candidates'])))
                    if replacement is None:
                        replacement = NamedTemporaryFile(dir=path.parent, prefix=f'.{path.name}.',
                                                         delete=False)
                        # Preserve any leading, already-upgraded records byte for byte.
                        with path.open('rb') as original:
                            for _ in range(row_count):
                                replacement.write(original.readline())
                    line = json.dumps(record, ensure_ascii=False).encode('utf-8') + b'\n'
                if replacement is not None:
                    replacement.write(line if line.endswith(b'\n') else line + b'\n')
                elif not line.endswith(b'\n'):
                    source.write(b'\n')
                row_count += 1
                yield record['row_id'], previous
        if replacement is not None:
            replacement.close()
            temporary = Path(replacement.name)
            temporary.chmod(path.stat().st_mode & 0o777)
            temporary.replace(path)
    finally:
        if replacement is not None:
            replacement.close()
            Path(replacement.name).unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--stage', choices=('pubchem', 'nmrexp'), default='nmrexp')
    p.add_argument('--diagnostic', choices=('baseline', 'oracle-bag', 'oracle-graph'), default='baseline',
                   help='baseline：原评估；oracle-bag：给定真实片段（beam不生效）；oracle-graph：给定真实分子图')
    p.add_argument('--split', choices=('valid', 'test'), default='test')
    p.add_argument('--out', type=Path, required=True)
    add_spectrum_arguments(p)
    add_search_arguments(p)
    p.add_argument('--limit', type=int, default=10000, help='实际符合cohort的评价分子数，0全部')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--log-every', type=int, default=100,
                   help='每 N 个分子打印并写入 .progress.log，0 表示仅记录最终汇总')
    p.add_argument('--threads', type=int, default=0)
    p.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    options = search_options(a, REFERENCE_SEARCH)
    if a.limit < 0:
        raise ValueError('--limit must be nonnegative')
    model, ck, root, _ = load_checkpoint(a.checkpoint, a.device, a.data)
    model.eval()
    spectrum = resolved_options(a, dict(SPECTRUM_DEFAULTS, metadata=ck.get('train_args', {}).get('metadata', False)))
    del ck
    config = dict(checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=digest(a.checkpoint),
                  data=str(root), manifest_sha256=digest(root / 'manifest.json'), stage=a.stage, split=a.split,
                  **spectrum, **options, metric='exact_2d_connectivity', diagnostic=a.diagnostic)
    config.update(sample_shuffle=True, sample_seed=42, sample_epoch=0)
    a.out.parent.mkdir(parents=True, exist_ok=True)

    if a.stage == 'pubchem':
        config.update(
            pubchem_sampling=SAMPLING_ID,
            sample_shuffle=True,
            sample_seed=EVAL_SEED,
            sample_epoch=0
        )

    config_path = a.out.with_suffix('.config.json')
    totals, seen = Counter(), set()
    resuming = a.out.exists()
    if resuming:
        if not a.resume:
            raise FileExistsError('输出已存在：使用--resume或换输出文件')
        previous = json.loads(config_path.read_text())
        # Outputs of the original evaluator used one implicit generation formula.
        for key in ('connect_normalization', 'score_mode', 'alpha', 'connect_weight'):
            previous.setdefault(key, LEGACY_SEARCH[key])
        previous.setdefault('manifest_sha256', config['manifest_sha256'])
        if previous != config:
            raise ValueError('续测配置必须一致')
    progress = ProgressLog(a.out.with_suffix('.progress.log'), a.log_every, resume=resuming)
    if resuming:
        for row_id, result in restore_metrics(a.out):
            if row_id in seen:
                raise ValueError(f'Duplicate row_id in evaluation output: {row_id}')
            seen.add(row_id)
            totals.update(result)
            if progress.due(len(seen), replay=True):
                progress.write(format_summary(mean_metrics(len(seen), totals, with_tomi=True)), echo=False)
    if a.limit and len(seen) > a.limit:
        raise ValueError('Existing evaluation contains more records than --limit')
    write_json(config_path, config)
    with a.out.open('a', encoding='utf-8') as f:
        for rows in batches(root, a.stage, a.split, size=1,
                            epoch=0, seed=42, shuffle=True):
            if a.limit and len(seen) >= a.limit:
                break
            row = rows[0]
            if row['row_id'] in seen or (a.stage != 'pubchem' and not eligible(row, spectrum['mode'], spectrum['cohort'])):
                continue
            x, candidates = predict_record(model, row, a.stage, options, spectrum, a.diagnostic)
            result = evaluate_one(row['smiles'], candidates, with_tomi=True)
            f.write(json.dumps({'row_id': row['row_id'], 'smiles': row['smiles'], 'input': x,
                                'candidates': candidates, 'metrics': result}, ensure_ascii=False)+'\n')
            f.flush()
            seen.add(row['row_id'])
            totals.update(result)
            if progress.due(len(seen)):
                progress.write(format_summary(mean_metrics(len(seen), totals, with_tomi=True)))
    if not seen:
        raise ValueError('No eligible records in the requested split and cohort')
    s = mean_metrics(len(seen), totals, with_tomi=True)
    write_json(a.out.with_suffix('.summary.json'), {**s, 'tomi_definition': TOMI_DESCRIPTION, 'config': config})
    progress.write(format_summary(s))


if __name__ == '__main__':
    main()
