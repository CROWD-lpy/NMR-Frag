"""01：保留完整峰注释，按分子划分；屏蔽预训练中的 NMRexp 留出分子。"""
import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from data_io import (normalize_smiles, parse_peaks, read_raw, read_pubchem, pick,
                     hash_id, number, write_json, new_dataset, ShardWriter, RECORD_SCHEMA, STRUCT_SCHEMA)


def random_split(mol_id, seed):
    u = int(hash_id(f'{seed}:{mol_id}')[:8], 16) / 2**32
    return 'train' if u < .855 else 'valid' if u < .9 else 'test'


def convert(nmrexp, pubchem, out, cid_mod=100, max_heavy=100, split_file=None, seed=42, limit=0):
    if cid_mod < 1 or max_heavy < 1 or limit < 0:
        raise ValueError('cid-mod and max-heavy must be positive; limit must be nonnegative')
    with new_dataset(out) as staged:
        return _convert(nmrexp, pubchem, staged, cid_mod, max_heavy, split_file, seed, limit)


def _convert(nmrexp, pubchem, out, cid_mod, max_heavy, split_file, seed, limit):
    split_map = {}
    if split_file:
        with open(split_file, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                split_map[normalize_smiles(pick(r, ('smiles', 'SMILES')))['mol_id']] = {'val': 'valid', 'validation': 'valid'}.get(r['split'], r['split'])
    db_path = out / '_join.sqlite'
    db = sqlite3.connect(db_path)
    db.execute('CREATE TABLE records (id TEXT PRIMARY KEY, mol_id TEXT, split TEXT, payload TEXT)')
    counts = Counter()
    priorities = {'train': 0, 'valid': 1, 'test': 2}
    explicit = False
    for i, (file_part, row_id, raw) in enumerate(read_raw(nmrexp)):
        if limit and i >= limit:
            break
        counts['NMRexp/read'] += 1
        try:
            info = normalize_smiles(pick(raw, ('smiles_canonical', 'SMILES', 'smiles', 'canonical_smiles')))
            if info['heavy_atoms'] > max_heavy:
                raise ValueError('heavy_too_large')
            part = split_map.get(info['mol_id']) or raw.get('split') or file_part
            part = {'val': 'valid', 'validation': 'valid'}.get(part, part)
            if part is not None and part not in priorities:
                raise ValueError('invalid_split')
            explicit |= part is not None
            if split_file and info['mol_id'] not in split_map:
                raise ValueError('not_in_split_file')
            part = part or random_split(info['mol_id'], seed)
            nucleus = str(raw.get('NMR_type', '')).replace(' ', '').replace('^', '')
            if nucleus and nucleus not in ('1HNMR', '13CNMR', '1H', '13C'):
                counts['NMRexp/other_nucleus'] += 1
                continue
            if nucleus:
                kind = 'h' if '1H' in nucleus else 'c'
                peaks = {kind: parse_peaks(raw.get('NMR_processed'), kind.upper())}
                # 同一 DOI、结构图位置的 H/C 才配对；避免跨论文合成一个虚构实验。
                location = '|'.join(str(raw.get(k, '')) for k in ('Filename', 'Page_in_file_mol', 'Location_in_page_mol'))
                row_id = hash_id(location + '|' + info['smiles']) if raw.get('Filename') else row_id
            else:
                peaks = {k: parse_peaks(pick(raw, keys), k.upper()) for k, keys in [
                    ('h', ('h_peaks', 'h_nmr_peaks', 'h_shifts', 'h_shift')),
                    ('c', ('c_peaks', 'c_nmr_peaks', 'c_shifts', 'c_shift'))]}
            if not any(peaks.values()):
                raise ValueError('empty_spectra')
            old = db.execute('SELECT payload, split FROM records WHERE id=?', (row_id,)).fetchone()
            row = json.loads(old[0]) if old else {**info, 'row_id': row_id, 'source': 'NMRexp', 'split': part,
                'h_peaks': [], 'c_peaks': []}
            if old and priorities[old[1]] > priorities[part]:
                part = old[1]
            for k, values in peaks.items():
                if values and not row.get(k + '_peaks'):
                    row[k + '_peaks'] = values
                    row[k + '_solvent'] = str(pick(raw, (k + '_solvent', k + '_nmr_solvent', 'NMR_solvent', 'solvent')) or '')
                    row[k + '_mhz'] = number(pick(raw, (k + '_mhz', k + '_nmr_frequency', 'NMR_frequency', 'frequency')))
                    row[k + '_raw'] = json.dumps(raw, ensure_ascii=False, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))
            db.execute('INSERT OR REPLACE INTO records VALUES (?,?,?,?)',
                       (row_id, info['mol_id'], part, json.dumps(row, ensure_ascii=False)))
        except (ValueError, TypeError, SyntaxError, RuntimeError) as e:
            counts['NMRexp/reject/' + str(e).split('\n')[0][:60]] += 1
        if i % 10000 == 0:
            db.commit()
    db.commit()
    mol_split = {}
    for mid, part in db.execute('SELECT DISTINCT mol_id, split FROM records'):
        if mid not in mol_split or priorities[part] > priorities[mol_split[mid]]:
            mol_split[mid] = part
    sink, structs = ShardWriter(out / 'records', RECORD_SCHEMA), ShardWriter(out / 'structures', STRUCT_SCHEMA)
    seen, buffers, sbuf = defaultdict(set), defaultdict(list), defaultdict(list)
    def emit(row):
        group = row['source'], row['split']
        buffers[group].append(row)
        if row['mol_id'] not in seen[group]:
            seen[group].add(row['mol_id'])
            sbuf[group].append(row)
        counts['/'.join(group)] += 1
        if len(buffers[group]) >= 1024:
            sink.write(group, buffers[group]); structs.write(group, sbuf[group])
            buffers[group].clear(); sbuf[group].clear()
    try:
        for mid, original, payload in db.execute('SELECT mol_id, split, payload FROM records ORDER BY id'):
            row = json.loads(payload)
            row['split'] = mol_split[mid]
            counts['split_reassigned'] += row['split'] != original
            emit(row)
        seen_pub = set()
        for i, (cid, smiles) in enumerate(read_pubchem(pubchem, cid_mod)):
            if limit and i >= limit:
                break
            counts['PubChem/read'] += 1
            try:
                info = normalize_smiles(smiles)
                mid = info['mol_id']
                if info['heavy_atoms'] > max_heavy:
                    raise ValueError('heavy_too_large')
                if mol_split.get(mid) in ('valid', 'test'):
                    counts['PubChem/excluded_NMRexp_holdout'] += 1
                    continue
                if mid in seen_pub:
                    continue
                seen_pub.add(mid)
                part = 'train' if mid in mol_split else random_split(mid, seed + 1)
                emit({**info, 'row_id': hash_id('PubChem:' + cid), 'source': 'PubChem', 'split': part,
                      'h_peaks': [], 'c_peaks': []})
            except (ValueError, RuntimeError) as e:
                counts['PubChem/reject/' + str(e).split('\n')[0][:60]] += 1
        for group in buffers:
            sink.write(group, buffers[group]); structs.write(group, sbuf[group])
    finally:
        sink.close(); structs.close(); db.close()
    db_path.unlink()
    if not mol_split and not seen_pub:
        raise ValueError('No usable records; check input files, split map and CID sampling')
    report = {'counts': dict(counts), 'seed': seed, 'max_heavy': max_heavy,
              'split_protocol': 'provided' if explicit else 'custom_molecule_hash_85.5_4.5_10',
              'pairing': 'provided paired rows; raw nucleus rows: DOI+structure-image location',
              'metric': '2D connectivity without stereochemistry'}
    write_json(out / 'manifest.json', report)
    print(json.dumps({k: v for k, v in counts.items() if k.count('/') == 1 and k.split('/')[-1] in ('train', 'valid', 'test')}, ensure_ascii=False))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nmrexp', required=True, type=Path)
    p.add_argument('--pubchem', required=True, type=Path)
    p.add_argument('--out', type=Path, default=Path('data/processed/uniform'))
    p.add_argument('--cid-mod', type=int, default=100)
    p.add_argument('--max-heavy', type=int, default=100)
    p.add_argument('--split-file', type=Path, help='CSV: smiles,split；对齐外部基准时使用')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=0, help='仅快速试跑；0为全部')
    a = p.parse_args()
    convert(a.nmrexp, a.pubchem, a.out, a.cid_mod, a.max_heavy, a.split_file, a.seed, a.limit)


if __name__ == '__main__':
    main()
