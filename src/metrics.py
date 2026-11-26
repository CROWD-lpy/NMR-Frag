"""Exact 2D hits and optional best-of-k ECFP4 Tanimoto similarity."""
from functools import lru_cache
from itertools import islice

KS = (1, 3, 5, 10)
HIT_KEYS = tuple(f'hit{k}' for k in KS)
RATE_KEYS = (*HIT_KEYS, 'oracle_pool', 'has_candidate')
TOMI_KEYS = tuple(f'tomi{k}' for k in KS)
TOMI_DESCRIPTION = ('Mean over targets of the highest Tanimoto similarity among the first k candidates. '
                    'Morgan bit fingerprint (radius 2, 2048 bits, no chirality). Empty pools score zero.')


@lru_cache(maxsize=1)
def _morgan_generator():
    from rdkit.Chem import rdFingerprintGenerator
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False, useBondTypes=True)


@lru_cache(maxsize=4096)
def _fingerprint(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot fingerprint invalid SMILES: {smiles!r}')
    return _morgan_generator().GetFingerprint(mol)


def topk_tomi(target, smiles):
    from rdkit import DataStructs
    top = list(islice(smiles, KS[-1]))
    values = DataStructs.BulkTanimotoSimilarity(_fingerprint(target), [_fingerprint(s) for s in top])
    return {f'tomi{k}': max(values[:k], default=0.) for k in KS}


def valid_unique_smiles(candidates):
    from chem import canonical
    seen = set()
    for c in candidates:
        try:
            s = canonical(c['smiles'] if isinstance(c, dict) else c)
        except (ValueError, RuntimeError):
            continue
        if s not in seen:
            seen.add(s)
            yield s


def evaluate_one(target, candidates, *, with_tomi=False):
    from chem import canonical
    target = canonical(target)
    smiles = list(valid_unique_smiles(candidates))
    result = {**{f'hit{k}': float(target in smiles[:k]) for k in KS},
              'oracle_pool': float(target in smiles), 'has_candidate': float(bool(smiles))}
    if with_tomi:
        result.update(topk_tomi(target, smiles))
    return result


def mean_metrics(n, totals, *, with_tomi=False):
    keys = (*RATE_KEYS, *TOMI_KEYS) if with_tomi else RATE_KEYS
    return {'n': n, **{key: totals[key] / n if n else 0. for key in keys}}


def format_summary(s):
    parts = [f'n={s["n"]}', *(f'{k}={s[k]:.2%}' for k in RATE_KEYS)]
    parts.extend(f'Tani@{k}={s[f"tomi{k}"]:.4f}' for k in KS if f'tomi{k}' in s)
    return '  '.join(parts)



class ProgressLog:
    """Mirror cumulative snapshots to stdout and disk, including a final partial interval."""
    def __init__(self, path, every, *, resume=False):
        if every < 0:
            raise ValueError('--log-every must be nonnegative')
        self.path = path
        self.every = every
        self.last_line = ''
        self.last_printed = ''
        if resume and path.exists():
            with path.open('r+b') as stream:
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line:
                        break
                    if not line.endswith(b'\n'):
                        stream.truncate(start)
                        break
                    if line.startswith(b'n='):
                        self.last_line = line.decode('utf-8').rstrip()
        else:
            path.write_text('', encoding='utf-8')
        self.resume_n = int(self.last_line.split()[0][2:]) if self.last_line else 0

    def due(self, n, *, replay=False):
        return self.every > 0 and n % self.every == 0 and (not replay or n > self.resume_n)

    def write(self, message, *, echo=True):
        if message != self.last_line:
            with self.path.open('a', encoding='utf-8') as stream:
                print(message, file=stream, flush=True)
            self.last_line = message
        if echo and message != self.last_printed:
            print(self.last_line, flush=True)
            self.last_printed = message
