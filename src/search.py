"""Fragment selection, port assembly and structure ranking for one decoding configuration."""
import argparse
import math
import numpy as np
import torch
from torch.nn import functional as F
from rdkit import Chem
from chem import formula_vector, graph_data
from data import condition as spectrum_condition
from model import add_log_probs


REFERENCE_SEARCH = dict(beam=64, connect_beam=16, pool_size=0, connect_normalization='pair-mask',
                        score_mode='joint', alpha=.5, connect_weight=.25, rank_mix=.9)
TRAIN_SEARCH = dict(REFERENCE_SEARCH, beam=16, connect_beam=4)
# Used for legacy validation on resume and for the original hard-negative mining protocol.
LEGACY_SEARCH = dict(beam=32, connect_beam=8, pool_size=128, connect_normalization='feasible',
                     score_mode='stage', alpha=1., connect_weight=1., rank_mix=.7)
LEGACY_TRAIN_SEARCH = dict(LEGACY_SEARCH, beam=16, connect_beam=4)


def add_search_arguments(parser, prefix=''):
    """Training uses --eval-* so inference weights cannot be confused with loss weights."""
    flag = '--' + prefix.replace('_', '-')
    parser.add_argument(flag + 'beam', type=int, help='fragment multiset beam width')
    parser.add_argument(flag + 'connect-beam', type=int, help='connection beam width per multiset')
    parser.add_argument(flag + 'pool-size', type=int, help='unique candidates to retain after ranking (0 = all)')
    parser.add_argument(flag + 'connect-normalization', choices=('feasible', 'pair-mask'))
    parser.add_argument(flag + 'score-mode', choices=('stage', 'joint'))
    parser.add_argument(flag + 'alpha', type=float, help='action-count normalization exponent')
    parser.add_argument(flag + 'connect-weight', type=float, help='Connect contribution to the generation score')
    parser.add_argument(flag + 'rank-mix', type=float, help='fraction of the standardized Rank score (0 = skip Rank)')


SPECTRUM_DEFAULTS = dict(mode='hc', cohort='paired', formula=True, annotations='full')


def add_spectrum_arguments(parser, prefix=''):
    flag = '--' + prefix.replace('_', '-')
    parser.add_argument(flag + 'mode', choices=('hc', 'h', 'c'))
    parser.add_argument(flag + 'cohort', choices=('paired', 'available'))
    parser.add_argument(flag + 'formula', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(flag + 'annotations', choices=('full', 'shift'))
    parser.add_argument(flag + 'metadata', action=argparse.BooleanOptionalAction, default=None)


def resolved_options(args, defaults, prefix='', saved=None):
    saved = saved or {}
    values = {}
    for key, default in defaults.items():
        name = prefix + key
        value = getattr(args, name)
        values[key] = saved.get(name, default) if value is None else value
    return values


def search_options(args, defaults, prefix='', saved=None):
    options = resolved_options(args, defaults, prefix, saved)
    if min(options['beam'], options['connect_beam']) < 1 or options['pool_size'] < 0:
        raise ValueError('Beam widths must be positive and pool size must be nonnegative')
    for key in ('alpha', 'connect_weight'):
        if not math.isfinite(options[key]) or options[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if not 0 <= options['rank_mix'] <= 1:
        raise ValueError('rank_mix must be in [0, 1]')
    return options


def repeat_context(ctx, n):
    return {k: v.expand(n, *v.shape[1:]) for k, v in ctx.items()}


@torch.no_grad()
def bag_beam(model, condition, ctx, width=32, max_fragments=64):
    frontier, finished = {(): 0.}, {}
    bank = model.keys()
    evidence = model.prepare_add_candidates([condition], ctx, bank)
    for depth in range(max_fragments + 1):
        if not frontier:
            break
        states = list(frontier)
        # evidence的batch维为1，评分时广播到各beam；整次搜索只读取一次条件。
        score, legal = model.add_logits([condition]*len(states), states,
                                       repeat_context(ctx, len(states)), bank,
                                       candidate_evidence=evidence)
        lp = add_log_probs(score, legal).cpu().numpy(); legal = legal.cpu().numpy()
        following = {}
        for i, ids in enumerate(states):
            if legal[i, -1]:
                value = frontier[ids] + float(lp[i, -1])
                finished[ids] = float(np.logaddexp(finished.get(ids, -np.inf), value))
            if depth == max_fragments:
                continue
            valid = np.flatnonzero(legal[i, :-1])
            k = min(len(valid), width*2)
            chosen = valid[np.argpartition(lp[i, valid], -k)[-k:]] if k else []
            for fid in chosen:
                nxt = tuple(sorted((*ids, int(fid))))
                value = frontier[ids] + float(lp[i, fid])
                following[nxt] = float(np.logaddexp(following.get(nxt, -np.inf), value))
        frontier = dict(sorted(following.items(), key=lambda x: (-x[1], x[0]))[:width])
    return sorted(finished.items(), key=lambda x: (-x[1], x[0]))[:width]


@torch.no_grad()
def assemble(model, bags, condition, ctx, width=8, batch_size=32, *,
             connect_normalization='feasible', score_mode='stage', alpha=1., connect_weight=1.):
    cat = model.cat
    # state = (bag编号，片段IDs，已知端点对，连接累积分数，bag分数)
    states = [(i, ids, (), 0., score) for i, (ids, score) in enumerate(bags)]
    pool = {}
    max_edges = max((sum(cat.port_totals[list(ids)])//2 for ids, _ in bags), default=0)
    for _ in range(int(max_edges)+1):
        active = []
        for state in states:
            bid, ids, edges, value, bag_score = state
            if np.any(cat.remaining(ids, edges)):
                active.append(state)
                continue
            result = cat.rebuild(ids, edges)
            if result is None:
                continue
            formula = condition.get('formula_counts')
            if formula is not None and not np.array_equal(formula_vector(result[1]), formula):
                continue
            # Compare terminal paths under the requested scoring rule before deduplication.
            na, nc = len(ids) + 1, len(edges)
            if score_mode == 'stage':
                prior = bag_score / na**alpha + connect_weight * value / max(1, nc)**alpha
            else:
                prior = (bag_score + connect_weight * value) / (na + nc)**alpha
            if not math.isfinite(prior):
                raise ValueError('Nonfinite generation score')
            smiles, mol_formula = result
            if smiles not in pool or prior > pool[smiles]['prior']:
                pool[smiles] = dict(smiles=smiles, formula=mol_formula, prior=prior,
                                    add_logp=bag_score, connect_logp=value,
                                    n_fragments=len(ids), n_connections=nc)
        if not active:
            break
        following = {}
        for start in range(0, len(active), batch_size):
            chunk = active[start:start+batch_size]
            logits = model.connection_logits([s[1] for s in chunk], [s[2] for s in chunk], repeat_context(ctx, len(chunk)))
            for state, score in zip(chunk, logits):
                bid, ids, known, old_score, bag_score = state
                mask = cat.pair_mask(ids, known); remaining = cat.remaining(ids, known)
                anchors = np.flatnonzero(remaining > 0)
                anchor = int(min(anchors, key=lambda u: (mask[u].sum()-remaining[u], u)))
                partners = [int(v) for v in np.flatnonzero(mask[anchor])
                            if cat.partial_feasible(ids, [*known, tuple(sorted((anchor, int(v))))])]
                if not partners:
                    continue
                if connect_normalization == 'feasible':
                    lp = F.log_softmax(score[anchor, partners].float(), -1)
                else:
                    allowed = torch.as_tensor(mask[anchor], device=score.device)
                    logits_row = score[anchor].float().masked_fill(~allowed, -torch.inf)
                    lp = F.log_softmax(logits_row, -1)[partners]
                lp = lp.cpu().numpy()
                for j in np.argsort(-lp, kind='stable')[:width]:
                    edge = tuple(sorted((anchor, partners[j])))
                    edges = tuple(sorted((*known, edge)))
                    value = old_score + float(lp[j])
                    key = bid, edges
                    if key not in following or value > following[key][3]:
                        following[key] = (bid, ids, edges, value, bag_score)
        groups = {}
        for state in following.values():
            groups.setdefault(state[0], []).append(state)
        states = [s for group in groups.values() for s in sorted(group, key=lambda x: (-x[3], x[2]))[:width]]
    return list(pool.values())


@torch.no_grad()
def generate(model, condition, graph=None, beam=32, connect_beam=8, max_fragments=64,
             pool_size=128, rank_mix=.7, *, connect_normalization='feasible', score_mode='stage',
             alpha=1., connect_weight=1., fragment_ids=None):
    ctx = model.encode_condition([condition], [graph] if graph is not None else None)
    if fragment_ids is None:
        bags = bag_beam(model, condition, ctx, beam, max_fragments)
    else:
        bags = [(tuple(fragment_ids), 0.)] if len(fragment_ids) else []
    candidates = assemble(model, bags, condition, ctx, connect_beam,
                          connect_normalization=connect_normalization, score_mode=score_mode,
                          alpha=alpha, connect_weight=connect_weight)
    if not candidates:
        return []
    if rank_mix and model.config.rank_weight:
        # Canonical order makes Rank batches agree with the reference scoring implementation.
        candidates.sort(key=lambda c: c['smiles'])
        for start in range(0, len(candidates), 32):
            chunk = candidates[start:start + 32]
            graphs = [graph_data(Chem.MolFromSmiles(c['smiles'])) for c in chunk]
            scores = model.score_graphs(graphs, repeat_context(ctx, len(chunk))).cpu().tolist()
            for candidate, score in zip(chunk, scores):
                if not math.isfinite(score):
                    raise ValueError('Nonfinite Rank score')
                candidate['match'] = score
        priors = standardize([c['prior'] for c in candidates])
        matches = standardize([c['match'] for c in candidates])
        for candidate, prior, match in zip(candidates, priors, matches):
            candidate.update(prior_z=prior, match_z=match, score=(1 - rank_mix) * prior + rank_mix * match)
    else:
        for candidate in candidates:
            candidate['score'] = candidate['prior']
    candidates.sort(key=lambda c: (-c['score'], c['smiles']))
    return candidates[:pool_size] if pool_size else candidates


def standardize(values):
    mean = math.fsum(values) / len(values)
    std = math.sqrt(math.fsum((v - mean)**2 for v in values) / len(values))
    return [(v - mean) / max(std, 1e-6) for v in values]


def predict_record(model, row, stage, search, spectrum, diagnostic='baseline'):
    inputs = spectrum_condition(row, spectrum['mode'], spectrum['formula'],
                                spectrum['annotations'], spectrum['metadata'])
    graph = None
    if stage == 'pubchem' or diagnostic == 'oracle-graph':
        graph = {'x': row['graph_x'], 'edges': row['graph_edges']}
    fragment_ids = None
    if diagnostic == 'oracle-bag':
        fragment_ids = row['fragment_ids'] if row['has_reference'] else []
    candidates = generate(model, inputs, graph, max_fragments=model.config.max_heavy,
                          fragment_ids=fragment_ids, **search)
    return inputs, candidates
