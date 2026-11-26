"""02：只从 train 学习拆分规则，保留高频片段并用原子模板兜底。"""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from data_io import iter_rows, write_json, new_dataset
from fragment_core import Fragmenter


def build(data, out, max_atoms=12, merges=256, mine=5000, min_count=1000, top_k=3000):
    if max_atoms < 1 or mine < 1 or min(merges, min_count, top_k) < 0:
        raise ValueError('Invalid fragment mining parameters')
    with new_dataset(out) as staged:
        return _build(data, staged, max_atoms, merges, mine, min_count, top_k)


def _build(data, out, max_atoms, merges, mine, min_count, top_k):
    sample = defaultdict(list)
    for row in iter_rows(Path(data) / 'structures'):
        if row['split'] == 'train' and len(sample[row['source']]) < mine:
            sample[row['source']].append(row['smiles'])
    model = Fragmenter(max_atoms, merges).fit([s for group in sample.values() for s in group])
    counts, seen = Counter(), set()
    for row in iter_rows(Path(data) / 'structures'):
        if row['split'] != 'train' or row['mol_id'] in seen:
            continue
        seen.add(row['mol_id'])
        state = model.replay(row['smiles'])
        counts.update(state.graph.port_token(n.atoms)[0] for n in state.roots())
    frequent = [t for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if n >= min_count]
    model.known_tokens = set(frequent[:top_k] if top_k else frequent)
    model.vocab_molecules = len(seen)
    model.freeze().save(out / 'tokenizer.json')
    write_json(out / 'summary.json', {'train_molecules': len(seen), 'rules': len(model.rules),
        'vocab_size': len(model.vocab), 'min_count': min_count, 'top_k': top_k,
        'atomic_backoff': True})
    print(f'train={len(seen):,}  rules={len(model.rules)}  vocab={len(model.vocab):,}')
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('data/processed/uniform'))
    p.add_argument('--out', type=Path, default=Path('data/processed/fragments'))
    p.add_argument('--max-atoms', type=int, default=12)
    p.add_argument('--merges', type=int, default=256)
    p.add_argument('--mine-per-source', type=int, default=5000)
    p.add_argument('--min-count', type=int, default=1000)
    p.add_argument('--top-k', type=int, default=3000)
    a = p.parse_args()
    build(a.data, a.out, a.max_atoms, a.merges, a.mine_per_source, a.min_count, a.top_k)


if __name__ == '__main__':
    main()
