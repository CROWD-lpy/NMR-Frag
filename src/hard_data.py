"""融合候选的schema与逐记录抽样；训练路径不解析SMILES。"""
import random
import pyarrow as pa
from chem import negative_graph
from data_io import hash_id

HARD_DATA_VERSION = 1
CANDIDATE_FIELD = pa.field('hard_candidates', pa.list_(pa.struct([
    ('smiles', pa.string()),
    ('graph_x', pa.list_(pa.list_(pa.float32()))),
    ('graph_edges', pa.list_(pa.list_(pa.int32()))),
])))


def sample_candidate_graphs(rows, epoch=0, seed=42, limit=4, *, local=False):
    """Align/Rank共享一次抽样；按row_id和task_epoch确定，续训可复现。"""
    if limit < 1:
        raise ValueError('hard-negatives必须>0')
    result = []
    for row in rows:
        rng = random.Random(int(hash_id(f'candidate:{seed}:{epoch}:{row["row_id"]}'), 16))
        if local:
            smiles = row['negative_smiles']
            sampled = rng.sample(smiles, min(limit, len(smiles)))
            result.append([g for s in sampled if (g := negative_graph(s)) is not None])
        else:
            pool = row['hard_candidates']
            chosen = rng.sample(pool, min(limit, len(pool)))
            result.append([{'x': c['graph_x'], 'edges': c['graph_edges']} for c in chosen])
    return result
