"""03：完整峰表 + 片段集合/端点连接；检查独立重建，保留未覆盖的评价分子。"""
import argparse
import json
import shutil
from collections import Counter
from functools import lru_cache
from pathlib import Path
import numpy as np
import pyarrow as pa
from rdkit import Chem
from chem import Catalog, VERSION, graph_data, negative_graph, formula_vector, digest
from data_io import RECORD_SCHEMA, ShardWriter, iter_rows, write_json, new_dataset
from fragment_core import Fragmenter

MODEL_FIELDS = [f for f in RECORD_SCHEMA if f.name not in ('h_raw', 'c_raw')]
SCHEMA = pa.schema([*MODEL_FIELDS,
    ('formula_counts', pa.list_(pa.int32())), ('has_reference', pa.bool_()),
    ('fragment_ids', pa.list_(pa.int32())), ('connections', pa.list_(pa.list_(pa.int32()))),
    ('graph_x', pa.list_(pa.list_(pa.float32()))), ('graph_edges', pa.list_(pa.list_(pa.int32()))),
    ('negative_smiles', pa.list_(pa.string()))])


def edge_alternatives(cat, ids, target, smiles, n=2, seed=42):
    """同片段、同分子式的错连结构：交换两条同阶键的端点，保持端口容量。"""
    if len(target) < 2 or not n:
        return []
    rng, gs = np.random.default_rng(seed), cat.layout(ids)['groups']
    found = set()
    for _ in range(48):
        i, j = rng.choice(len(target), 2, replace=False)
        a, b = target[i]; c, d = target[j]
        if len({a, b, c, d}) < 4 or gs[a]['order'] != gs[c]['order']:
            continue
        if rng.random() < .5:
            c, d = d, c
        replacement = [sorted((a, d)), sorted((c, b))]
        edges = sorted([e for k, e in enumerate(target) if k not in (i, j)] + replacement)
        result = cat.rebuild(ids, edges)
        if result and result[0] != smiles and negative_graph(result[0]) is not None:
            found.add(result[0])
            if len(found) >= n:
                break
    return sorted(found)


def prepare(data, fragments, out, hard_negatives=2):
    if hard_negatives < 0:
        raise ValueError('--hard-negatives must be nonnegative')
    with new_dataset(out) as staged:
        return _prepare(data, fragments, staged, hard_negatives)


def _prepare(data, fragments, out, hard_negatives):
    tokenizer = Path(fragments) / 'tokenizer.json'
    model, cat = Fragmenter.load(tokenizer), Catalog(tokenizer)
    shutil.copyfile(tokenizer, out / 'tokenizer.json')
    cat.save(out / 'catalog.pt')
    @lru_cache(maxsize=4096)
    def labels(smiles):
        graph = graph_data(Chem.MolFromSmiles(smiles))
        part = model.decompose(smiles)
        ids = [f['fragment_id'] for f in part['fragments']]
        if not part['covered']:
            return graph, [], [], False, []
        layout = cat.layout(ids)
        edges = []
        for e in part['connections']:
            ends = [layout['group_offsets'][e[side]] + cat.port_to_group[ids[e[side]]][e[side + '_port']]
                    for side in ('left', 'right')]
            edges.append(sorted(ends))
        edges.sort()
        reference = cat.rebuild(ids, edges) == (smiles, part['formula'])
        reference &= bool(np.array_equal(cat.formulas[ids].sum(0), formula_vector(part['formula'])))
        negatives = edge_alternatives(cat, ids, edges, smiles, hard_negatives,
                                     int(part['mol_id'][:8], 16)) if reference else []
        return graph, ids, edges, reference, negatives
    counts, covered, rejected, max_fragments, max_heavy = Counter(), Counter(), Counter(), 0, 0
    writer, buffers = ShardWriter(out / 'examples', SCHEMA), {}
    try:
        for row in iter_rows(Path(data) / 'records', columns=[f.name for f in MODEL_FIELDS]):
            group = row['source'], row['split']
            graph, ids, edges, reference, negatives = labels(row['smiles'])
            key = '/'.join(group)
            if group[1] == 'train' and not reference:
                rejected[key] += 1
                continue
            record = {**row, 'formula_counts': formula_vector(row['formula']).tolist(),
                      'graph_x': graph['x'], 'graph_edges': graph['edges'], 'has_reference': reference,
                      'fragment_ids': ids if reference else [], 'connections': edges if reference else [],
                      'negative_smiles': negatives}
            buffers.setdefault(group, []).append(record)
            counts[key] += 1; covered[key] += reference
            max_fragments = max(max_fragments, len(ids)); max_heavy = max(max_heavy, row['heavy_atoms'])
            if len(buffers[group]) >= 1024:
                writer.write(group, buffers[group]); buffers[group].clear()
        for group, rows in buffers.items():
            writer.write(group, rows)
    finally:
        writer.close()
    manifest = {'version': VERSION, 'tokenizer_sha256': digest(tokenizer), 'rows': dict(counts),
                'reference_rows': dict(covered), 'train_rejected': dict(rejected),
                'vocab_size': len(cat), 'max_fragments': max_fragments, 'max_heavy': max_heavy,
                'input_manifest': json.loads((Path(data) / 'manifest.json').read_text())}
    write_json(out / 'manifest.json', manifest)
    print('  '.join(f'{k}={v:,} (covered={covered[k]/v:.1%})' for k, v in sorted(counts.items())))
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/processed/uniform'))
    p.add_argument('--fragments', type=Path, default=Path('data/processed/fragments'))
    p.add_argument('--out', type=Path, default=Path('data/processed/overall_3000'))
    p.add_argument('--hard-negatives', type=int, default=2, help='每分子缓存错连结构数；0关闭')
    a = p.parse_args()
    prepare(a.data, a.fragments, a.out, a.hard_negatives)


if __name__ == '__main__':
    main()
