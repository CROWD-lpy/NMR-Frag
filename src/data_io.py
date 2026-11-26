"""NMRexp 原始 CSV/Parquet 与 NMRPeak LMDB 的完整信号读取。"""
import ast
import csv
import gzip
import hashlib
import io
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from contextlib import contextmanager
from tempfile import TemporaryDirectory
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

ELEMENTS = {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I', 'B', 'Si', 'Se'}
PEAK = pa.struct([('ppm', pa.float32()), ('lo', pa.float32()), ('hi', pa.float32()),
                  ('integral', pa.float32()), ('mult', pa.string()), ('j', pa.list_(pa.float32()))])
STRUCT_FIELDS = [('mol_id', pa.string()), ('source', pa.string()), ('split', pa.string()),
                 ('smiles', pa.string()), ('formula', pa.string()), ('heavy_atoms', pa.int32())]
STRUCT_SCHEMA = pa.schema(STRUCT_FIELDS)
RECORD_SCHEMA = pa.schema([('row_id', pa.string()), *STRUCT_FIELDS,
    ('h_peaks', pa.list_(PEAK)), ('c_peaks', pa.list_(PEAK)),
    ('h_solvent', pa.string()), ('c_solvent', pa.string()),
    ('h_mhz', pa.float32()), ('c_mhz', pa.float32()),
    ('h_raw', pa.string()), ('c_raw', pa.string())])
MULTS = ['<missing>', '<other>', 's', 'd', 't', 'q', 'm', 'br', 'dd', 'dt', 'td',
         'ddd', 'ddt', 'dq', 'qd', 'tt', 'quint', 'sept', 'dddd', 'ddq', 'ddp', 'dqd', 'bs', 'bd']
SOLVENTS = ['<missing>', '<other>', 'cdcl3', 'dmso', 'cd3od', 'd2o', 'acetone',
            'c6d6', 'cd3cn', 'cd2cl2', 'thf', 'dmf', 'c5d5n']


def hash_id(text):
    return hashlib.sha256(str(text).encode()).hexdigest()[:32]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


@contextmanager
def new_dataset(path):
    """Publish the directory only after the complete data stage succeeds."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Choose a new output directory: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=f'.{path.name}-') as temp:
        staged = Path(temp) / 'dataset'
        staged.mkdir()
        yield staged
        staged.rename(path)


def pick(row, keys):
    return next((row[k] for k in keys if k in row and row[k] is not None), None)


def number(x):
    if x is None:
        return None
    if isinstance(x, str):
        match = re.search(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', x)
        if not match:
            return None
        x = match[0]
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def unpack(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, str):
        if not x.strip() or x.lower() in ('none', 'nan', 'null'):
            return None
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return ast.literal_eval(x)
    return x


def normalize_smiles(raw):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(str(raw or ''))
    if mol is None:
        raise ValueError('invalid_smiles')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('multi_component')
    if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() or a.GetIsotope() for a in mol.GetAtoms()):
        raise ValueError('charge_radical_isotope')
    mol = Chem.RemoveHs(mol)
    if not mol.GetNumAtoms() or any(a.GetSymbol() not in ELEMENTS for a in mol.GetAtoms()):
        raise ValueError('unsupported_element')
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    Chem.RemoveStereochemistry(mol)
    smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
    return {'mol_id': hash_id(smiles), 'smiles': smiles,
            'formula': rdMolDescriptors.CalcMolFormula(mol), 'heavy_atoms': mol.GetNumHeavyAtoms()}


def mult_id(x):
    s = re.sub(r'[ ._-]', '', str(x or '').lower())
    s = {'brs': 'bs', 'brd': 'bd', 'broad': 'br', 'p': 'quint', 'hept': 'sept',
         'quintet': 'quint', 'septet': 'sept'}.get(s, s)
    return 0 if not s else MULTS.index(s) if s in MULTS else 1


def solvent_id(x):
    s = re.sub(r'[\s_{}-]', '', str(x or '').lower())
    aliases = {'dmso': 'dmso', 'acetone': 'acetone', 'thf': 'thf', 'dmf': 'dmf'}
    for prefix, name in aliases.items():
        if s.startswith(prefix):
            s = name
    return 0 if not s else SOLVENTS.index(s) if s in SOLVENTS else 1


def parse_peaks(raw, nucleus):
    """H：(multiplicity, J列表, '3H', delta1, delta2)。C：(delta,multiplicity,J)。"""
    items = unpack(raw)
    if items is None:
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, (list, tuple)):
        items = [items]
    result = []
    for p in items:
        p = unpack(p) if isinstance(p, np.ndarray) else p
        integral = mult = js = lo = hi = None
        if isinstance(p, dict):
            ppm = number(pick(p, ('ppm', 'centroid', 'delta', 'delta (ppm)', 'shift')))
            lo = number(pick(p, ('lo', 'rangeMin', 'range_min', 'delta_min')))
            hi = number(pick(p, ('hi', 'rangeMax', 'range_max', 'delta_max')))
            integral = number(pick(p, ('integral', 'nH', 'nh', 'integration')))
            mult = pick(p, ('mult', 'category', 'multiplicity', 'multiplicity_raw'))
            js = pick(p, ('j', 'j_values', 'jvalue', 'jValue', 'J', 'J (Hz)', 'coupling', 'couplings'))
        elif isinstance(p, (list, tuple)):
            if nucleus == 'H' and len(p) == 5 and isinstance(p[0], str):
                mult, js, integral, lo, hi = p
                lo, hi, integral = number(lo), number(hi), number(integral)
                ppm = None
            elif nucleus == 'C':
                ppm = number(p[0]); mult = p[1] if len(p) > 1 else None
                js = p[2] if len(p) > 2 else None
            else:
                raise ValueError('unknown_H_peak_format')
        else:
            ppm = number(p)
        if lo is not None and hi is not None:
            lo, hi = sorted((lo, hi))
            ppm = (lo + hi) / 2 if ppm is None else ppm
        if ppm is None:
            raise ValueError('invalid_peak_' + nucleus)
        bounds = (-3, 17) if nucleus == 'H' else (-50, 300)
        if not bounds[0] <= ppm <= bounds[1] or any(v is not None and not bounds[0] <= v <= bounds[1] for v in (lo, hi)):
            raise ValueError('shift_outlier_' + nucleus)
        if integral is not None and integral <= 0:
            raise ValueError('invalid_integral')
        if isinstance(js, str):
            s = js.strip()
            if s.startswith('['):
                js = unpack(s)
            elif s.lower() in ('', 'none', 'nan', 'null'):
                js = None
            else:
                # NMRPeak LMDB 用 '8.1_2.3_'；单峰/多重峰无J常写作 '_'。
                js = [x for x in re.split('[,;_]', s.replace('J=', '').replace('J =', '')) if x.strip()]
        if js is not None:
            js = list(js) if isinstance(js, (list, tuple, np.ndarray)) else [js]
            parsed = [number(x) for x in js]
            if any(x is None for x in parsed):
                raise ValueError('invalid_J')
            js = sorted(parsed)
        result.append({'ppm': ppm, 'lo': lo, 'hi': hi, 'integral': integral,
                       'mult': None if mult is None else str(mult), 'j': js})
    return sorted(result, key=lambda p: p['ppm'])


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if int(np.__version__.split('.')[0]) < 2:
            module = module.replace('numpy._core', 'numpy.core')
        return super().find_class(module, name)


def file_split(path):
    for part in reversed(Path(path).parts):
        s = part.lower().split('.')[0]
        for name, aliases in [('train', ('train',)), ('valid', ('val', 'valid', 'validation')), ('test', ('test',))]:
            if any(re.search(r'(^|[_-])' + x + r'([_-]|$)', s) for x in aliases):
                return name
    return None


def read_raw(root):
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    suffixes = ('.lmdb', '.parquet', '.csv', '.jsonl')
    if root.is_file() or (root / 'data.mdb').is_file():
        files = [root]
    else:
        files = sorted({p.parent if p.name == 'data.mdb' else p for p in root.rglob('*')
                        if p.name == 'data.mdb' or (p.is_file() and p.suffix.lower() in suffixes)})
    if not files:
        raise ValueError(f'No NMRexp data files in {root}')
    for path in files:
        split = file_split(path)
        if path.suffix.lower() == '.lmdb' or path.is_dir():
            import lmdb
            env = lmdb.open(str(path), subdir=path.is_dir(), readonly=True, lock=False)
            try:
                with env.begin() as txn:
                    for key, blob in txn.cursor():
                        if not key.startswith(b'__'):
                            yield split, hash_id(str(path) + ':' + key.decode()), CompatibleUnpickler(io.BytesIO(blob)).load()
            finally:
                env.close()
        elif path.suffix.lower() == '.parquet':
            i = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
                for row in batch.to_pylist():
                    yield split, hash_id(str(path) + ':' + str(i)), row
                    i += 1
        elif path.suffix.lower() == '.csv':
            with path.open(encoding='utf-8-sig', errors='replace', newline='') as f:
                for i, row in enumerate(csv.DictReader(f)):
                    yield split, hash_id(str(path) + ':' + str(i)), row
        elif path.suffix.lower() == '.jsonl':
            with path.open(encoding='utf-8') as f:
                for i, line in enumerate(f):
                    yield split, hash_id(str(path) + ':' + str(i)), json.loads(line)

        else:
            raise ValueError(f'Unsupported NMRexp file: {path}')


def read_pubchem(path, cid_mod=100):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as f:
        for line in f:
            fields = line.split(None, 1)
            if len(fields) == 2 and fields[0].isdigit() and int(fields[0]) % cid_mod == 0:
                yield fields[0], fields[1].strip()


class ShardWriter:
    def __init__(self, root, schema):
        self.root, self.schema = Path(root), schema
        self.writers, self.counts = {}, Counter()
    def write(self, group, rows):
        if not rows:
            return
        part = self.counts[group] // 65536
        key = (*group, part)
        if key not in self.writers:
            if self.writers.get(group):
                self.writers.pop(group).close()
            folder = self.root / f'source={group[0]}' / f'split={group[1]}'
            folder.mkdir(parents=True, exist_ok=True)
            self.writers[group] = pq.ParquetWriter(folder / f'part-{part:05d}.parquet', self.schema, compression='zstd')
            self.writers[key] = True
        self.writers[group].write_table(pa.Table.from_pylist(rows, schema=self.schema), row_group_size=1024)
        self.counts[group] += len(rows)
    def close(self):
        for key, writer in self.writers.items():
            if len(key) == 2:
                writer.close()


def iter_rows(root, columns=None):
    for batch in ds.dataset(str(root), format='parquet').to_batches(batch_size=4096, columns=columns):
        yield from batch.to_pylist()
