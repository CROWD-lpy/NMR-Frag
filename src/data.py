"""流式读取、固定种子与显式缺失条件；真实标签不进入无分子式生成。"""
import itertools
import json
import random
from pathlib import Path
from collections import Counter
import pyarrow.dataset as ds
import torch
from chem import Catalog, VERSION, digest
from data_io import hash_id

SOURCES = {'pubchem': 'PubChem', 'nmrexp': 'NMRexp'}


def open_data(root):
    root = Path(root).resolve()
    info = json.loads((root / 'manifest.json').read_text())
    if info['version'] != VERSION or digest(root / 'tokenizer.json') != info['tokenizer_sha256']:
        raise ValueError('V3数据/词表不匹配，请重新处理')
    catalog = Catalog(root / 'catalog.pt')
    if catalog.tokenizer_sha256 != info['tokenizer_sha256']:
        raise ValueError('catalog.pt and tokenizer.json belong to different vocabularies')
    return root, info, catalog


def eligible(row, mode, cohort='available'):
    h, c = bool(row['h_peaks']), bool(row['c_peaks'])
    if mode == 'hc' or cohort == 'paired':
        return h and c
    return h if mode == 'h' else c


def seed_for(row, epoch=0, seed=42):
    return int(hash_id(f'{seed}:{epoch}:{row["row_id"]}')[:16], 16)


SAMPLING_ID = 'pubchem-global-arrow-v1'
EVAL_SEED = 42


def batches(root, stage, split, size=64, epoch=0, seed=42, shuffle=False, limit=0,
            reference_only=False, include_candidates=False):
    """PubChem uses a global permutation. NMRexp streams shuffled Parquet blocks."""
    if size < 1 or limit < 0:
        raise ValueError('Batch size must be positive and limit must be nonnegative')
    folder = Path(root) / 'examples' / f'source={SOURCES[stage]}' / f'split={split}'
    dataset = ds.dataset(str(folder), format='parquet')
    excluded = {'h_raw', 'c_raw'}
    if not include_candidates:
        excluded.update(('hard_candidates', 'hard_candidate_source'))
    columns = [name for name in dataset.schema.names if name not in excluded]
    files = sorted(dataset.get_fragments(), key=lambda f: f.path)
    row_filter = ds.field('has_reference') if reference_only else None
    if stage == 'pubchem':
        table = ds.dataset([f.path for f in files], format='parquet').to_table(
            columns=columns, filter=row_filter).combine_chunks()
        order = list(range(len(table)))
        random.Random(seed + epoch if split == 'train' else EVAL_SEED).shuffle(order)
        stop = min(limit, len(order)) if limit else len(order)
        for start in range(0, stop, size):
            yield table.take(order[start:min(start + size, stop)]).to_pylist()
        return
    rng = random.Random(seed + epoch)
    if shuffle:
        rng.shuffle(files)
    block_size = max(size, 256) if include_candidates and 'hard_candidates' in columns else 4096

    def rows():
        for file in files:
            for block in file.to_batches(batch_size=block_size, columns=columns, filter=row_filter):
                records = block.to_pylist()
                if shuffle:
                    rng.shuffle(records)
                yield from records

    stream = itertools.islice(rows(), limit or None)
    while batch := list(itertools.islice(stream, size)):
        yield batch


def condition(row, mode='hc', formula=True, annotations='full', metadata=False):
    """生成器只获得本字典；没有 smiles/graph/真实片段数/隐藏分子式。"""
    result = {'h_peaks': row.get('h_peaks', []) if mode in ('h', 'hc') else [],
              'c_peaks': row.get('c_peaks', []) if mode in ('c', 'hc') else [],
              'formula_counts': row['formula_counts'] if formula else None}
    for k in ('h', 'c'):
        result[k + '_solvent'] = row.get(k + '_solvent') if metadata and result[k + '_peaks'] else None
        result[k + '_mhz'] = row.get(k + '_mhz') if metadata and result[k + '_peaks'] else None
        if annotations == 'shift':
            result[k + '_peaks'] = [dict(p, integral=None, mult=None, j=None, lo=None, hi=None)
                                    for p in result[k + '_peaks']]
    return result


def augment(rows, epoch, seed=42, modality_drop=.3, formula_drop=.25, field_drop=.1, peak_drop=0., metadata=False):
    outputs = []
    for row in rows:
        rng = random.Random(seed_for(row, epoch, seed))
        x = condition(row, formula=rng.random() >= formula_drop, metadata=metadata)
        if x['h_peaks'] and x['c_peaks'] and rng.random() < modality_drop:
            x[rng.choice(('h_peaks', 'c_peaks'))] = []
        for k in ('h', 'c'):
            keep = []
            for peak in x[k + '_peaks']:
                if rng.random() < peak_drop:
                    continue
                p = dict(peak)
                for field in ('integral', 'mult', 'j'):
                    if rng.random() < field_drop:
                        p[field] = None
                keep.append(p)
            # 不训练无谱样本；只剩一个核种时至少留一个信号。
            if x[k + '_peaks'] and not keep:
                keep = [dict(rng.choice(x[k + '_peaks']))]
            x[k + '_peaks'] = keep
            if not keep:
                x[k + '_solvent'] = x[k + '_mhz'] = None
        outputs.append(x)
    return outputs


def sample_tasks(rows, epoch=0, seed=42, early=False, connect_empty=.3):
    if not 0 <= connect_empty <= 1:
        raise ValueError('connect_empty必须在[0,1]')
    tasks = []
    for row in rows:
        rng = random.Random(seed_for(row, epoch, seed))
        order = list(row['fragment_ids']); rng.shuffle(order)
        if early and len(order) > 1:
            prob = rng.random()
            if prob < .10:
                n = 0
            elif prob < .20:
                n = len(order)
            else:
                # 仅NMRexp训练启用early：80%均匀覆盖1..N-1；不额外偏向短prefix。
                n = rng.randrange(1, len(order))
        else:
            n = rng.randrange(len(order) + 1)
        keep = 0. if rng.random() < connect_empty else rng.random() * .9
        tasks.append({'prefix': order[:n], 'remaining': Counter(order[n:]),
                      'known': [e for e in row['connections'] if rng.random() < keep],
                      'target': row['connections'], 'seed': rng.randrange(2**31)})
    return tasks


def pack_graphs(graphs, device):
    xs, edges, orders, owner, offsets = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        offsets.append(offset)
        xs.extend(g['x']); owner.extend([i] * len(g['x']))
        for u, v, order in g['edges']:
            edges.extend([[offset+u, offset+v], [offset+v, offset+u]])
            orders.extend([order, order])
        offset += len(g['x'])
    return (torch.tensor(xs, dtype=torch.float32, device=device),
            torch.tensor(edges, dtype=torch.long, device=device).reshape(-1, 2).T,
            torch.tensor(orders, dtype=torch.long, device=device),
            torch.tensor(owner, dtype=torch.long, device=device), offsets)


def padded_tokens(flat, lengths):
    n = max(1, max(lengths, default=0))
    x = flat.new_zeros((len(lengths), n, flat.shape[-1]))
    mask = torch.zeros((len(lengths), n), dtype=torch.bool, device=flat.device)
    start = 0
    for i, k in enumerate(lengths):
        x[i, :k] = flat[start:start+k]; mask[i, :k] = True
        start += k
    return x, mask
