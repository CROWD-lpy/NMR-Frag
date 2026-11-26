"""Merge verified, offline NMRexp training candidates into a new step-03 dataset."""
import argparse
import json
import os
import shutil
import sqlite3
import tempfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rdkit import Chem
from chem import VERSION, digest, graph_data
from data_io import normalize_smiles, write_json
from hard_data import HARD_DATA_VERSION, CANDIDATE_FIELD


def merge(data, hard_cache, out, batch_size=128, link_unchanged=False, require_all=False):
    root, cache, out = Path(data).resolve(), Path(hard_cache).resolve(), Path(out).resolve()
    if batch_size < 1:
        raise ValueError('batch-size必须>0')
    if out.exists():
        raise FileExistsError('请使用不存在的新输出目录；原数据不会被覆盖')
    if root in out.parents or out in root.parents:
        raise ValueError('输入、输出目录不能互相包含')
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    source_sha, tokenizer_sha = digest(root/'manifest.json'), digest(root/'tokenizer.json')
    if manifest['version'] != VERSION or manifest['tokenizer_sha256'] != tokenizer_sha:
        raise ValueError('原数据版本/词表不匹配')
    if manifest.get('hard_candidates'):
        raise ValueError('请使用原03数据进行融合，不能重复融合已处理的数据')
    config = json.loads(cache.with_suffix('.config.json').read_text(encoding='utf-8'))
    if (config.get('stage'), config.get('split'), config.get('mode'),
            config.get('formula'), config.get('annotations')) != ('nmrexp', 'train', 'hc', True, 'full'):
        raise ValueError('Expected a formula-conditioned NMRexp training cache from step 07')
    if config.get('cohort', 'paired') not in ('paired', 'available'):
        raise ValueError('Unknown training cohort in cache')
    if config.get('data_manifest_sha256') != source_sha or config.get('tokenizer_sha256') != tokenizer_sha:
        raise ValueError('困难缓存与原数据manifest/词表不同')

    out.parent.mkdir(parents=True, exist_ok=True)
    # SQLite索引与流式Parquet避免把全量谱图/候选图同时驻留内存。
    with tempfile.TemporaryDirectory(prefix='merge-hard-', dir=out.parent) as temp:
        work = Path(temp)
        dest = work/'dataset'
        dest.mkdir()
        db = sqlite3.connect(work/'index.sqlite')
        try:
            db.executescript('''
                CREATE TABLE training(row_id TEXT PRIMARY KEY, mol_id TEXT, smiles TEXT, formula TEXT);
                CREATE TABLE candidates(row_id TEXT PRIMARY KEY, mol_id TEXT, graphs TEXT);
                CREATE INDEX by_molecule ON candidates(mol_id, row_id);
                CREATE TABLE features(smiles TEXT PRIMARY KEY, formula TEXT, graph TEXT);
            ''')
            train = ds.dataset(str(root/'examples/source=NMRexp/split=train'), format='parquet')
            for block in train.to_batches(columns=['row_id', 'mol_id', 'smiles', 'formula'],
                                           filter=ds.field('has_reference'), batch_size=8192):
                try:
                    db.executemany('INSERT INTO training VALUES (?, ?, ?, ?)',
                                   [(r['row_id'], r['mol_id'], r['smiles'], r['formula']) for r in block.to_pylist()])
                except sqlite3.IntegrityError as exc:
                    raise ValueError('训练集row_id重复') from exc
            db.commit()
            print('已建立训练记录索引，开始一次性校验候选并生成图特征…', flush=True)
            rejected = Counter()

            @lru_cache(maxsize=512)
            def features(raw):
                info = normalize_smiles(raw)
                s = info['smiles']
                cached = db.execute('SELECT graph FROM features WHERE smiles=?', (s,)).fetchone()
                if cached:
                    return info, json.loads(cached[0])
                graph = graph_data(Chem.MolFromSmiles(s))
                value = dict(smiles=s, graph_x=graph['x'], graph_edges=graph['edges'])
                db.execute('INSERT INTO features VALUES (?, ?, ?)',
                           (s, info['formula'], json.dumps(value)))
                return info, value

            def validated(target, formula, candidates, origin):
                target = normalize_smiles(target)['smiles']
                result, seen = [], set()
                for raw in candidates:
                    if isinstance(raw, dict):
                        raw = raw.get('smiles')
                    try:
                        info, graph = features(raw)
                    except (ValueError, TypeError, RuntimeError):
                        rejected[origin+'_invalid'] += 1
                        continue
                    s = info['smiles']
                    if s == target or s in seen or info['formula'] != formula:
                        rejected[origin+'_target_duplicate_or_formula'] += 1
                        continue
                    seen.add(s)
                    result.append(graph)
                return result

            cache_rows = 0
            with cache.open(encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    try:
                        record = json.loads(line)
                        rid, mid, target = record['row_id'], record['mol_id'], record['smiles']
                        negatives = record['negative_smiles']
                        if not isinstance(negatives, list):
                            raise ValueError('negative_smiles不是列表')
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(f'困难缓存第{line_no}行损坏') from exc
                    row = db.execute('SELECT mol_id, smiles, formula FROM training WHERE row_id=?', (rid,)).fetchone()
                    if row is None:
                        raise ValueError(f'候选记录不属于当前NMRexp训练集：{rid}')
                    if mid != row[0] or normalize_smiles(target)['smiles'] != normalize_smiles(row[1])['smiles']:
                        raise ValueError(f'候选的目标结构不匹配：{rid}')
                    graphs = validated(target, row[2], negatives, 'hard')
                    try:
                        db.execute('INSERT INTO candidates VALUES (?, ?, ?)', (rid, mid, json.dumps(graphs)))
                    except sqlite3.IntegrityError as exc:
                        raise ValueError(f'困难缓存row_id重复：{rid}') from exc
                    cache_rows += 1
                    if cache_rows % 1000 == 0:
                        db.commit()
                        print(f'已处理 {cache_rows:,} 条候选记录', flush=True)
            db.commit()

            @lru_cache(maxsize=128)
            def molecule_candidates(mid):
                # 07每个分子只生成一条代表谱图；该分子的其他谱记录共享同一结构候选池。
                item = db.execute("SELECT graphs FROM candidates WHERE mol_id=? AND graphs!='[]' ORDER BY row_id LIMIT 1",
                                  (mid,)).fetchone()
                return json.loads(item[0]) if item else []

            counts = Counter()
            # PubChem与词表/目录均原样复制；可选择硬链接以避免复制大规模PubChem文件。
            for path in sorted(root.rglob('*')):
                relative = path.relative_to(root)
                output = dest/relative
                if path.is_dir():
                    output.mkdir(parents=True, exist_ok=True)
                    continue
                if relative == Path('manifest.json'):
                    continue
                is_nmr = relative.parts[:2] == ('examples', 'source=NMRexp') and path.suffix == '.parquet'
                if not is_nmr:
                    if link_unchanged:
                        try:
                            os.link(path, output)
                        except OSError:
                            shutil.copy2(path, output)
                    else:
                        shutil.copy2(path, output)
                    continue
                source = pq.ParquetFile(path)
                schema = source.schema_arrow.append(CANDIDATE_FIELD).append(pa.field('hard_candidate_source', pa.string()))
                with pq.ParquetWriter(output, schema, compression='zstd') as writer:
                    for block in source.iter_batches(batch_size=batch_size):
                        rows = block.to_pylist()
                        for row in rows:
                            split, rid = row['split'], row['row_id']
                            pool, origin = [], 'none'
                            if row['has_reference'] and split == 'train':
                                own = db.execute('SELECT graphs FROM candidates WHERE row_id=?', (rid,)).fetchone()
                                pool = json.loads(own[0]) if own else molecule_candidates(row['mol_id'])
                                if pool:
                                    origin = 'hard_row' if own else 'hard_molecule'
                            elif row['has_reference']:
                                # 不把训练候选复制到验证/测试；只预计算其03已有负例的原始特征。
                                pool = validated(row['smiles'], row['formula'], row.get('negative_smiles') or [], 'heldout')
                                if pool:
                                    origin = 'local_negative'
                            row['hard_candidates'], row['hard_candidate_source'] = pool, origin
                            counts[f'{split}/rows'] += 1
                            counts[f'{split}/{origin}'] += 1
                            counts[f'{split}/candidates'] += len(pool)
                            if split == 'train' and row['has_reference'] and not pool:
                                counts['train/without_candidates'] += 1
                        writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=batch_size)
                db.commit()
                print(f'已融合 {relative}', flush=True)
            if not counts['train/candidates']:
                raise ValueError('训练集没有任何可用的困难候选；未发布输出数据')
            if require_all and counts['train/without_candidates']:
                raise ValueError(f"仍有 {counts['train/without_candidates']} 条训练记录没有困难候选；未发布输出数据")
            manifest['hard_candidates'] = dict(
                version=HARD_DATA_VERSION, field='hard_candidates', raw_graph_features=True,
                checkpoint_sha256=config['checkpoint_sha256'], cohort=config.get('cohort', 'paired'),
                source_manifest_sha256=source_sha, cache_sha256=digest(cache), cache_rows=cache_rows,
                tokenizer_sha256=tokenizer_sha, counts=dict(counts), rejected=dict(rejected),
                train_empty='skip_align_rank', heldout='original_local_negatives_only')
            write_json(dest/'manifest.json', manifest)
            write_json(dest/'hard_candidates_report.json', manifest['hard_candidates'])
            dest.rename(out)
        finally:
            db.close()
    print(json.dumps(manifest['hard_candidates'], ensure_ascii=False, indent=2), flush=True)
    print(f'完成：{out}', flush=True)
    return manifest['hard_candidates']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True, help='原03数据目录，与07使用的一致')
    p.add_argument('--hard-cache', type=Path, required=True, help='07输出的JSONL，旁边须有.config.json')
    p.add_argument('--out', type=Path, required=True, help='新的融合数据目录')
    p.add_argument('--batch-size', type=int, default=128, help='图特征转换块大小，控制内存')
    p.add_argument('--link-unchanged', action='store_true', help='硬链接未修改文件，失败则复制；之后请勿原位改写这些文件')
    p.add_argument('--require-all', action='store_true', help='有参考标签的训练记录存在空候选池时失败')
    args = p.parse_args()
    merge(args.data, args.hard_cache, args.out, args.batch_size, args.link_unchanged, args.require_all)


if __name__ == '__main__':
    main()
