"""V6余弦评分：共享条件/集合读取，余弦加证据修正，CE监督。"""
import math
import random
from copy import deepcopy
from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from candidate_add import CandidateAddHead, CandidateConditionAttention
from chem import ATOM_DIM, ELEMENTS, negative_graph
from data import pack_graphs, padded_tokens
from data_io import MULTS, SOLVENTS, mult_id, solvent_id

# 仅新NMRexp run启用；保持命令行简洁，实际权重写入config.json。
NMR_AUX_WEIGHTS = dict(presence=.2, count=.1, align=.1)
MODEL_SHAPE_FIELDS = ('hidden', 'heads', 'graph_layers', 'peak_layers', 'set_layers',
                      'dynamic_layers', 'candidate_hidden', 'candidate_heads',
                      'candidate_layers')

@dataclass
class Config:
    hidden: int = 512
    heads: int = 8
    graph_layers: int = 4
    peak_layers: int = 3
    set_layers: int = 2
    dynamic_layers: int = 2
    dropout: float = .1
    connect_weight: float = 1.
    rank_weight: float = .3
    symmetry_trials: int = 4
    max_heavy: int = 100
    fragment_repr: str = 'descriptor'
    freeze_chem: bool = False
    auxiliary_tasks: bool = False
    graph_alignment: bool = False
    align_temperature: float = .1
    candidate_hidden: int = 512
    candidate_heads: int = 8
    candidate_layers: int = 2
    # 旧检查点缺省False以保持原预测；04_train的新训练显式开启并写入检查点。
    stable_add: bool = False

    def __post_init__(self):
        if any(getattr(self, name) < 1 for name in MODEL_SHAPE_FIELDS) or self.max_heavy < 1:
            raise ValueError('维度、头数、层数和max_heavy必须为正整数')
        if self.hidden % self.heads or self.candidate_hidden % self.candidate_heads:
            raise ValueError('hidden/candidate-hidden必须分别能被heads/candidate-heads整除')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout必须在[0,1)')

def mlp(a, b, h):
    return nn.Sequential(nn.Linear(a, h), nn.GELU(), nn.Linear(h, b))


def add_log_probs(scores, legal):
    """训练/生成/诊断共用严格mask；死路返回全-inf，不对全-inf直接做softmax。"""
    logits = scores.float().masked_fill(~legal, -torch.inf)
    logits = logits.masked_fill(~legal.any(-1, keepdim=True), 0.)
    return F.log_softmax(logits, -1).masked_fill(~legal, -torch.inf)


class MessageBlock(nn.Module):
    def __init__(self, h, dropout=0.):
        super().__init__()
        self.bond = nn.Embedding(5, h)
        self.message = mlp(h * 2, h, h)
        self.update = mlp(h * 2, h, h * 2)
        self.norm = nn.LayerNorm(h)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edges, orders):
        aggregate = torch.zeros_like(x)
        degree = x.new_zeros((len(x), 1))
        if edges.numel():
            source, target = edges
            message = self.message(torch.cat((x[source], self.bond(orders)), -1)).to(x.dtype)
            aggregate.index_add_(0, target, message)
            degree.index_add_(0, target, torch.ones_like(target, dtype=x.dtype).unsqueeze(1))
        aggregate = aggregate / degree.clamp_min(1)
        return self.norm(x + self.drop(self.update(torch.cat((x, aggregate), -1))))


class GraphEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c.hidden
        self.input = mlp(ATOM_DIM, h, h)
        self.layers = nn.ModuleList(MessageBlock(h, c.dropout) for _ in range(c.graph_layers))
        self.readout = mlp(3 * h + 1, h, 2 * h)

    def forward(self, graphs, device):
        x, edges, orders, owner, offsets = pack_graphs(graphs, device)
        x = self.input(x)
        for layer in self.layers:
            x = layer(x, edges, orders)
        n = len(graphs)
        total, square = x.new_zeros((n, x.shape[-1])), x.new_zeros((n, x.shape[-1]))
        total.index_add_(0, owner, x)
        square.index_add_(0, owner, x.square())
        counts = torch.bincount(owner, minlength=n).to(x.dtype).unsqueeze(-1).clamp_min(1)
        global_x = self.readout(torch.cat((total / counts, total / 16, square / counts, counts.log()), -1))
        return global_x, x, offsets


class GraphTeacher(nn.Module):
    """起始检查点的图条件编码器快照；不共享参数、不参与反向传播。"""
    def __init__(self, chemistry, projection):
        super().__init__()
        self.chem = deepcopy(chemistry)
        self.projection = deepcopy(projection)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, graphs, device):
        with torch.autocast(device_type=device.type, enabled=False):
            g, _, _ = self.chem(graphs, device)
            return self.projection(g).float()


class SpectrumEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        h = c.hidden
        self.register_buffer('centers', torch.linspace(0, 1, 48))
        self.register_buffer('frequencies', 2 ** torch.arange(8, dtype=torch.float32))
        # 物理ppm上的密集RBF与多周期Fourier；H/C各自学习数值投影。
        self.shift_nuclei = nn.ModuleList([
            NumericShift(h, -3., 20., .05, (.05, .25), (.05, .1, .2, .5, 1., 2., 5., 10.)),
            NumericShift(h, -50., 350., 1., (1., 5.), (.5, 1., 2., 5., 10., 20., 50., 100.))])
        self.j_value = mlp(2, h, h)
        self.mult = nn.Embedding(len(MULTS), h)
        self.nucleus = nn.Embedding(2, h)
        self.solvent = nn.Embedding(len(SOLVENTS), h)
        self.meta = mlp(3, h, h)
        self.formula = mlp(len(ELEMENTS) + 1, h, h)
        self.cls = nn.Parameter(torch.randn(1, 1, h) * .02)
        self.modality = nn.Embedding(4, h)
        block = nn.TransformerEncoderLayer(h, c.heads, 4*h, c.dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, c.peak_layers, enable_nested_tensor=False)
        self.readout = mlp(2*h + 2, h, 2*h)

    def forward(self, rows, device):
        features, mults, kinds, solvents, meta, js, lengths, counts = [], [], [], [], [], [], [], []
        for row in rows:
            n = 0
            counts.append([len(row.get('h_peaks', [])), len(row.get('c_peaks', []))])
            for kind, k in enumerate(('h', 'c')):
                for p in row.get(k + '_peaks', []):
                    scale, origin = (20., -3.) if kind == 0 else (350., -50.)
                    v = (p['ppm'] - origin) / scale
                    lo, hi = p.get('lo'), p.get('hi')
                    width = (hi-lo)/scale if lo is not None and hi is not None else 0.
                    integral, j = p.get('integral'), p.get('j')
                    features.append([v, width, (lo-origin)/scale if lo is not None else v,
                        (hi-origin)/scale if hi is not None else v,
                        math.log1p(integral or 0), float(integral is not None),
                        float(lo is not None and hi is not None), float(j is not None),
                        math.log1p(len(j or [])), float(p.get('mult') is not None)])
                    js.append(j or [])
                    mults.append(mult_id(p.get('mult'))); kinds.append(kind)
                    solvents.append(solvent_id(row.get(k + '_solvent')))
                    mhz = row.get(k + '_mhz')
                    meta.append([(mhz or 0)/600, float(mhz is not None), float(bool(row.get(k + '_solvent')))])
                    n += 1
            lengths.append(n)
        known = [r.get('formula_counts') is not None for r in rows]
        formula = torch.tensor([(r.get('formula_counts') or [0]*len(ELEMENTS)) + [int(v)]
                                for r, v in zip(rows, known)], device=device, dtype=torch.float32)
        formula[:, :-1] /= 20
        bos = self.formula(formula).unsqueeze(1)
        if features:
            f = torch.tensor(features, device=device, dtype=torch.float32)
            rbf = torch.exp(-.5 * ((f[:, :1]-self.centers)/.025).square())
            fourier = 2*math.pi*f[:, :1]*self.frequencies
            base = torch.cat((rbf, fourier.sin(), fourier.cos(), f), -1)
            kinds_tensor = torch.tensor(kinds, device=device)
            x = bos.new_zeros((len(base), bos.shape[-1]))
            for kind, numeric in enumerate(self.shift_nuclei):
                index = torch.nonzero(kinds_tensor == kind, as_tuple=True)[0]
                if len(index):
                    x = x.index_copy(0, index, numeric(base[index], f[index, :1]))
            # 每个 J 都参与集合池化；不截断成固定4个，不把缺失与空J列表混为一谈。
            jflat = [v for group in js for v in group]
            jfeatures = torch.tensor([[v/30, math.log1p(abs(v))/5] for v in jflat], device=device, dtype=torch.float32).reshape(-1, 2)
            jtokens, jmask = padded_tokens(self.j_value(jfeatures), [len(g) for g in js])
            jw = jmask.to(x.dtype).unsqueeze(-1)
            jembed = (jtokens*jw).sum(1)/jw.sum(1).clamp_min(1)
            x = x + jembed + self.mult(torch.tensor(mults, device=device)) + self.nucleus(torch.tensor(kinds, device=device))
            x = x + self.solvent(torch.tensor(solvents, device=device)) + self.meta(torch.tensor(meta, device=device))
        else:
            x = bos.new_zeros((0, bos.shape[-1]))
        x, mask = padded_tokens(x, lengths)
        modes = torch.tensor([int(n[0]>0) + 2*int(n[1]>0) for n in counts], device=device)
        cls = self.cls.expand(len(rows), -1, -1) + self.modality(modes)[:, None]
        tokens = torch.cat((cls, bos, x), 1)
        mask = torch.cat((torch.ones((len(rows), 2), dtype=torch.bool, device=device), mask), 1)
        tokens = self.encoder(tokens, src_key_padding_mask=~mask)
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        mean = (tokens*weights).sum(1)/weights.sum(1)
        count = torch.tensor(counts, device=device, dtype=tokens.dtype).log1p()/4
        z = self.readout(torch.cat((tokens[:, 0], mean, count), -1))
        return z, tokens, mask


class NumericShift(nn.Module):
    """保留原74维连续特征，增加核种专用、以ppm为单位的局部与周期特征。"""
    def __init__(self, hidden, origin, scale, spacing, widths, periods):
        super().__init__()
        self.origin, self.scale = origin, scale
        self.register_buffer('local_centers', torch.linspace(origin, origin+scale, round(scale/spacing)+1))
        self.register_buffer('widths', torch.tensor(widths, dtype=torch.float32))
        self.register_buffer('periods', torch.tensor(periods, dtype=torch.float32))
        extra = len(self.local_centers)*len(widths) + 2*len(periods)
        self.proj = mlp(74+extra, hidden, hidden)

    def forward(self, base, normalized_ppm):
        ppm = normalized_ppm*self.scale + self.origin
        distance = ppm[:, :, None] - self.local_centers[None, :, None]
        local = torch.exp(-.5*(distance/self.widths).square()).flatten(1)
        phase = 2*math.pi*(ppm-self.origin)/self.periods
        return self.proj(torch.cat((base, local, phase.sin(), phase.cos()), -1))


class Model(nn.Module):
    def __init__(self, cat, config=None):
        super().__init__()
        self.cat, self.config = cat, config or Config()
        c, h = self.config, self.config.hidden
        self.chem = GraphEncoder(c)
        self.spectrum = SpectrumEncoder(c)
        self.structure_projection = mlp(h, h, h)
        self.fragment_id = nn.Embedding(len(cat), h)
        descriptors = []
        for g, formula, groups in zip(cat.graphs, cat.formulas, cat.groups):
            x = np.asarray(g['x'], dtype=np.float32)
            bonds = [sum(e[2] == k for e in g['edges'])/8 for k in (1, 2, 3, 4)]
            ports = [sum(p['capacity'] for p in groups if p['order'] == k)/8 for k in (1, 2, 3)]
            descriptors.append([*x.mean(0), *(x.sum(0)/16), *(formula/20), *bonds, *ports])
        self.register_buffer('descriptors', torch.tensor(np.asarray(descriptors), dtype=torch.float32), persistent=False)
        self.fragment_feature = nn.Linear(self.descriptors.shape[-1], h)
        self.start = nn.Parameter(torch.zeros(1, 1, h))
        block = nn.TransformerDecoderLayer(h, c.heads, 2*h, c.dropout, batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerDecoder(block, c.set_layers)
        self.add_context = nn.MultiheadAttention(h, c.heads, c.dropout, batch_first=True)
        self.add_query = mlp(3*h + len(ELEMENTS) + 2, h, 2*h)
        self.stop = nn.Linear(h, 1)
        self.dynamic_input = nn.Linear(3, h)
        self.dynamic = nn.ModuleList(MessageBlock(h, c.dropout) for _ in range(c.dynamic_layers))
        self.order = nn.Embedding(4, h)
        self.endpoint = mlp(3*h+2, h, 2*h)
        self.port_context = nn.MultiheadAttention(h, c.heads, c.dropout, batch_first=True)
        self.port_norm = nn.LayerNorm(h)
        self.pair_projection = nn.Linear(h, h, bias=False)
        self.pair_unary = nn.Linear(h, 1)
        self.pair_geometry = mlp(2, 1, 32)
        self.match_context = nn.MultiheadAttention(h, c.heads, c.dropout, batch_first=True)
        self.match_head = mlp(5*h, 1, 2*h)
        self.register_buffer('formulas', torch.tensor(cat.formulas, dtype=torch.float32), persistent=False)
        self._all_ids = list(range(len(cat)))
        self._key_bank = None
        self._graph_cache = {}
        self._frozen_fragment_cache = None
        self._eval_fragment_cache = None
        self.fragment_cache_batch_size = 512
        self.fragment_graph = nn.Linear(h, h, bias=False)
        self.configure_fragments(c.fragment_repr, c.freeze_chem)
        self.candidate_condition = CandidateConditionAttention(c)
        self.candidate_add = CandidateAddHead(c)
        if c.auxiliary_tasks:
            self._make_auxiliary_heads()
        if c.graph_alignment:
            if not c.auxiliary_tasks:
                raise ValueError('图条件对齐需要辅助任务配置')
            self.graph_teacher = GraphTeacher(self.chem, self.structure_projection)

    def _make_auxiliary_heads(self):
        h = self.config.hidden
        self.presence_query = mlp(h, h, h).to(self.device)
        self.presence_bias = nn.Parameter(torch.zeros(len(self.cat), device=self.device))
        # 每个片段至少含一个重原子，上限使用数据的max_heavy，避免截断真实数量。
        self.count_head = mlp(h, self.config.max_heavy + 1, h).to(self.device)
        self.remaining_peak_query = nn.Linear(h, self.config.candidate_hidden).to(self.device)
        self.remaining_count = nn.Sequential(nn.Linear(2, 16), nn.GELU(),
                                            nn.Linear(16, 1)).to(self.device)
        nn.init.zeros_(self.remaining_count[-1].weight)
        nn.init.zeros_(self.remaining_count[-1].bias)
        self.remaining_gain = nn.Parameter(torch.zeros((), device=self.device))

    def enable_auxiliary(self, align=False):
        """切换NMRexp时增加辅助任务，并保存预训练图教师快照。"""
        if not self.config.auxiliary_tasks:
            self.config.auxiliary_tasks = True
            self._make_auxiliary_heads()
        if align and not self.config.graph_alignment:
            self.config.graph_alignment = True
            self.graph_teacher = GraphTeacher(self.chem, self.structure_projection)
        self.train(self.training)

    @property
    def device(self):
        return next(self.parameters()).device

    def train(self, mode=True):
        self._key_bank = None
        self._graph_cache = {}
        self._eval_fragment_cache = None
        super().train(mode)
        if self.config.freeze_chem:
            self.chem.eval()  # 固定dropout，冻结片段缓存才始终有效。
        if self.config.graph_alignment:
            self.graph_teacher.eval()
        return self

    def clear_fragment_cache(self):
        self._key_bank = None
        self._graph_cache = {}
        self._frozen_fragment_cache = None
        self._eval_fragment_cache = None

    def _apply(self, fn, recurse=True):
        self.clear_fragment_cache()  # .to()/.half()后不能复用旧设备或旧精度的缓存。
        return super()._apply(fn, recurse=recurse)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.clear_fragment_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def configure_fragments(self, fragment_repr=None, freeze_chem=None):
        mode = self.config.fragment_repr if fragment_repr is None else fragment_repr
        frozen = self.config.freeze_chem if freeze_chem is None else freeze_chem
        if mode not in ('descriptor', 'gnn', 'hybrid'):
            raise ValueError('fragment_repr应为descriptor/gnn/hybrid')
        self.config.fragment_repr, self.config.freeze_chem = mode, bool(frozen)
        self.chem.requires_grad_(not frozen)
        for p in self.chem.parameters():
            p.grad = None
        self.chem.train(self.training and not frozen)
        self.fragment_feature.requires_grad_(mode != 'gnn')
        self.fragment_graph.requires_grad_(mode != 'descriptor')
        self.clear_fragment_cache()

    def _encode_fragment_bank(self, batch_size=None):
        """分块编码全词表；训练时保留梯度，由本次loss局部持有计算图。"""
        batch_size = self.fragment_cache_batch_size if batch_size is None else batch_size
        if batch_size < 1:
            raise ValueError('片段缓存batch_size必须大于0')
        globals_, atoms, lengths = [], [], []
        for start in range(0, len(self.cat), batch_size):
            graphs = self.cat.graphs[start:start+batch_size]
            g, x, _ = self.chem(graphs, self.device)
            globals_.append(g); atoms.append(x)
            lengths.extend(len(a['x']) for a in graphs)
        offsets = np.cumsum([0, *lengths]).tolist()
        return torch.cat(globals_), torch.cat(atoms), offsets

    @torch.no_grad()
    def build_fragment_cache(self, batch_size=None):
        """冻结时分批缓存全词表的全局/原子表示；缓存不写入检查点。"""
        if not self.config.freeze_chem or any(p.requires_grad for p in self.chem.parameters()):
            raise RuntimeError('只有冻结的化学GNN可以建立训练用片段缓存')
        if self._frozen_fragment_cache is not None:
            return
        self.chem.eval()
        # 不让首次调用所处的AMP环境决定永久缓存精度。
        with torch.autocast(device_type=self.device.type, enabled=False):
            self._frozen_fragment_cache = self._encode_fragment_bank(batch_size)

    def fragment_bank(self):
        if self.config.freeze_chem:
            self.build_fragment_cache()
            return self._frozen_fragment_cache
        if self.training:
            return self._encode_fragment_bank()
        # eval期间权重固定，可复用；train()/eval()/加载权重/换设备均会清除。
        if self._eval_fragment_cache is None:
            with torch.no_grad(), torch.autocast(device_type=self.device.type, enabled=False):
                self._eval_fragment_cache = self._encode_fragment_bank()
        return self._eval_fragment_cache

    def keys(self, ids=None, fragment_bank=None):
        if not self.training and self._key_bank is not None:
            bank = self._key_bank
        else:
            mode = self.config.fragment_repr
            vectors = self.fragment_id.weight
            if mode in ('descriptor', 'hybrid'):
                vectors = vectors + self.fragment_feature(self.descriptors)
            if mode in ('gnn', 'hybrid'):
                bank = self.fragment_bank() if fragment_bank is None else fragment_bank
                vectors = vectors + self.fragment_graph(bank[0])
            bank = F.normalize(vectors, dim=-1)
            if not self.training:
                self._key_bank = bank.detach()
        return bank if ids is None else bank[torch.tensor(ids, device=self.device, dtype=torch.long)]

    def fragment_encoding(self, ids, fragment_bank=None):
        if not ids:
            return self.fragment_id.weight.new_zeros((0, self.config.hidden)), []
        if fragment_bank is None and (self.config.freeze_chem or
                (not self.training and self.config.fragment_repr in ('gnn', 'hybrid'))):
            fragment_bank = self.fragment_bank()
        if fragment_bank is not None:
            g, x, offsets = fragment_bank
            index = torch.tensor(ids, device=self.device, dtype=torch.long)
            return g[index], [x[offsets[f]:offsets[f+1]] for f in ids]
        unique = sorted(set(ids))
        if not self.training:
            missing = [f for f in unique if f not in self._graph_cache]
            if missing:
                g, x, offsets = self.chem([self.cat.graphs[f] for f in missing], self.device)
                for i, f in enumerate(missing):
                    self._graph_cache[f] = (g[i].detach(), x[offsets[i]:offsets[i]+len(self.cat.graphs[f]['x'])].detach())
            return torch.stack([self._graph_cache[f][0] for f in ids]), [self._graph_cache[f][1] for f in ids]
        g, x, offsets = self.chem([self.cat.graphs[f] for f in unique], self.device)
        remap = {f: i for i, f in enumerate(unique)}
        chunks = {f: x[offsets[i]:offsets[i]+len(self.cat.graphs[f]['x'])] for f, i in remap.items()}
        return g[torch.tensor([remap[f] for f in ids], device=self.device)], [chunks[f] for f in ids]

    def encode_condition(self, inputs, graphs=None):
        if graphs is not None:
            g, x, _ = self.chem(graphs, self.device)
            z = self.structure_projection(g)
            memory, mask = padded_tokens(x, [len(g['x']) for g in graphs])
        else:
            z, memory, mask = self.spectrum(inputs, self.device)
        memory = torch.cat((z[:, None], memory), 1)
        mask = torch.cat((torch.ones((len(inputs), 1), device=self.device, dtype=torch.bool), mask), 1)
        return {'z': z, 'memory': memory, 'mask': mask}

    def prepare_add_candidates(self, inputs, context, keys):
        """条件证据仅依赖当前memory与词表；推理时跨步/beam复用。"""
        return self.candidate_condition(keys, context['memory'], context['mask'])

    def remaining_logits(self, q, prefixes, bank, evidence):
        selected = bank.new_zeros((len(prefixes), len(bank)))
        owners = [i for i, ids in enumerate(prefixes) for _ in ids]
        flat = [f for ids in prefixes for f in ids]
        if flat:
            selected.index_put_((torch.tensor(owners, device=self.device),
                                 torch.tensor(flat, device=self.device)),
                                selected.new_ones(len(flat)), accumulate=True)
        count_features = torch.stack((selected.log1p(), (selected > 0).to(selected.dtype)), -1)
        logits = self.presence_query(q) @ bank.T + self.presence_bias
        matched = torch.matmul(evidence, self.remaining_peak_query(q).unsqueeze(-1)).squeeze(-1)
        return (logits + matched/math.sqrt(self.config.candidate_hidden)
                + self.remaining_count(count_features).squeeze(-1)).float()

    def add_logits(self, inputs, prefixes, context, keys=None, candidate_evidence=None,
                   return_presence=False):
        bank = self.keys() if keys is None else keys
        flat = [f for ids in prefixes for f in ids]
        x = bank[torch.tensor(flat, device=self.device, dtype=torch.long)] if flat else bank.new_zeros((0, self.config.hidden))
        x, mask = padded_tokens(x, [len(ids) for ids in prefixes])
        x = torch.cat((self.start.expand(len(inputs), -1, -1), x), 1)
        mask = torch.cat((torch.ones((len(inputs), 1), dtype=torch.bool, device=self.device), mask), 1)
        # 只规范Add读取的条件，不改变Connect/Rank/Count/Align的输入。
        memory = context['memory']
        if self.config.stable_add:
            memory = F.layer_norm(memory.float(), (memory.shape[-1],)).to(memory.dtype)
        # 无位置编码、无因果mask；每层先集合自注意力，再读取全部条件token。
        with sdpa_kernel(SDPBackend.MATH):
            x = self.set_encoder(x, memory, tgt_key_padding_mask=~mask,
                                 memory_key_padding_mask=~context['mask'])
        w = mask.to(x.dtype).unsqueeze(-1)
        total = (x*w).sum(1); mean = total/w.sum(1)
        with sdpa_kernel(SDPBackend.MATH):
            attention, _ = self.add_context(mean[:, None], memory, memory,
                                           key_padding_mask=~context['mask'], need_weights=False)
        used = np.asarray([self.cat.formulas[list(ids)].sum(0) if ids else np.zeros(len(ELEMENTS)) for ids in prefixes])
        known = np.asarray([r.get('formula_counts') is not None for r in inputs])
        remainder = np.asarray([np.asarray(r['formula_counts']) - u if k else u for r, u, k in zip(inputs, used, known)])
        info = np.concatenate((remainder/20, known[:, None], np.asarray([[len(ids)/16] for ids in prefixes])), 1)
        q = self.add_query(torch.cat((mean, total/16, attention[:, 0], torch.tensor(info, device=self.device, dtype=mean.dtype)), -1))
        if self.config.stable_add:
            # 单位RMS保持q方向，统一约束证据点积、Presence和STOP的输入尺度。
            # 仅余弦主项归一化不能约束这些共享分支；float32兼容AMP。
            q = F.normalize(q.float(), dim=-1, eps=1e-6) * math.sqrt(self.config.hidden)
        if candidate_evidence is None:
            candidate_evidence = self.prepare_add_candidates(inputs, context, bank)
        logits = self.candidate_add(bank, candidate_evidence, q, x, mask, context['z'])
        remaining = None
        if self.config.auxiliary_tasks:
            remaining = self.remaining_logits(q, prefixes, bank, candidate_evidence)
        if remaining is not None:
            # 剩余存在性只软修正分数，幅度不超过4；不作为硬筛选条件。
            logits = logits.float() + self.remaining_gain.tanh()*remaining.clamp(-4., 4.)
        logits = torch.cat((logits.float(), self.stop(q).float()), -1)
        legal = torch.ones_like(logits, dtype=torch.bool)
        for i, (r, ids, rem, k) in enumerate(zip(inputs, prefixes, remainder, known)):
            if k:
                fits = (self.cat.formulas <= rem).all(-1)
                heavy = [j for j, e in enumerate(ELEMENTS) if e != 'H']
                exact = (self.cat.formulas == rem).all(-1)
                heavy_exact = (self.cat.formulas[:, heavy] == rem[heavy]).all(-1)
                fits &= ~heavy_exact | exact
                maxh = self.cat.max_h_per_element
                fits &= (rem[1] - self.cat.formulas[:, 1]) <= rem[heavy] @ maxh - self.cat.formulas[:, heavy] @ maxh
                if not any(r['formula_counts'][j] for j, e in enumerate(ELEMENTS) if e not in ('C', 'H', 'N', 'O')):
                    f = r['formula_counts']; dbe = f[0] + 1 + (f[2]-f[1])/2
                    fits &= np.asarray(self.cat.unsaturation) + sum(self.cat.unsaturation[f] for f in ids) <= dbe + 1e-5
                fits &= (self.cat.port_totals > 0) | ((not ids) & exact)
                for j in np.flatnonzero(fits & exact):
                    fits[j] &= self.cat.can_finish_bag([*ids, int(j)])
                stop = not np.any(rem) and self.cat.can_finish_bag(ids)
            else:
                # 只有已选片段的信息和全局上限；完全不访问隐藏分子式。
                heavy = [j for j, e in enumerate(ELEMENTS) if e != 'H']
                fits = self.cat.formulas[:, heavy].sum(1) + used[i, heavy].sum() <= self.config.max_heavy
                if ids:
                    fits &= self.cat.port_totals > 0
                    if any(self.cat.port_totals[f] == 0 for f in ids):
                        fits[:] = False
                stop = self.cat.can_finish_bag(ids)
            legal[i, :-1] = torch.tensor(fits, device=self.device)
            legal[i, -1] = stop
        result = logits, legal  # 所有概率调用统一交给add_log_probs施加严格mask。
        return (*result, remaining) if return_presence else result
    def connection_logits(self, bags, known_edges, context, fragment_bank=None):
        """批量更新真实原子的当前结构，再为端点对打分；没有端口位置编码。"""
        all_ids = [f for ids in bags for f in ids]
        global_x, atom_chunks = self.fragment_encoding(all_ids, fragment_bank)
        atom_x = torch.cat(atom_chunks)
        dyn = np.zeros((len(atom_x), 3), dtype=np.float32)
        edges, orders, endpoints, lengths, geometries = [], [], [], [], []
        atom_base, frag_base = 0, 0
        for ids, known in zip(bags, known_edges):
            layout = self.cat.layout(ids)
            gs = layout["groups"]
            remaining = self.cat.remaining(ids, known)
            bonds = list(layout["internal"])
            bonds.extend([[gs[u]["atom"], gs[v]["atom"], gs[u]["order"]] for u, v in known])
            adjacent = [[] for _ in range(layout["n_atoms"])]
            for u, v, order in bonds:
                edges.extend([[atom_base + u, atom_base + v], [atom_base + v, atom_base + u]])
                orders.extend([order, order])
                adjacent[u].append(v)
                adjacent[v].append(u)
            for g, r in zip(gs, remaining):
                dyn[atom_base + g["atom"], g["order"] - 1] += r / 4
                endpoints.append((atom_base + g["atom"], frag_base + g["instance"],
                                  g["order"], g["capacity"] / 4, r / 4))
            # 有无路径和路径长度只描述结构，不含任意实例编号。
            geom = np.zeros((len(gs), len(gs), 2), dtype=np.float32)
            for u, g in enumerate(gs):
                distance = {g["atom"]: 0}
                queue = [g["atom"]]
                for node in queue:
                    for nxt in adjacent[node]:
                        if nxt not in distance:
                            distance[nxt] = distance[node] + 1
                            queue.append(nxt)
                for v, other in enumerate(gs):
                    if other["atom"] in distance:
                        geom[u, v] = [1., min(distance[other["atom"]], 20) / 20]
            geometries.append(geom)
            lengths.append(len(gs))
            atom_base += layout["n_atoms"]
            frag_base += len(ids)
        x = atom_x + self.dynamic_input(torch.tensor(dyn, device=self.device))
        edge = torch.tensor(edges, device=self.device, dtype=torch.long).reshape(-1, 2).T
        bond = torch.tensor(orders, device=self.device, dtype=torch.long)
        for layer in self.dynamic:
            x = layer(x, edge, bond)
        if endpoints:
            index = torch.tensor([e[:3] for e in endpoints], device=self.device, dtype=torch.long)
            count = torch.tensor([e[3:] for e in endpoints], device=self.device, dtype=torch.float32)
            fid = torch.tensor(all_ids, device=self.device)
            frag = global_x + self.fragment_id(fid)
            port = self.endpoint(torch.cat((x[index[:, 0]], frag[index[:, 1]], self.order(index[:, 2]), count), -1))
        else:
            port = x.new_zeros((0, self.config.hidden))
        port, valid = padded_tokens(port, lengths)
        # 历史非有限梯度集中于该注意力的输入投影及上游，局部使用math后端。
        with sdpa_kernel(SDPBackend.MATH):
            attended, _ = self.port_context(port, context["memory"], context["memory"],
                                            key_padding_mask=~context["mask"], need_weights=False)
        port = self.port_norm(port + attended)
        q = self.pair_projection(port)
        raw = q @ port.transpose(1, 2)
        logits = (raw + raw.transpose(1, 2)) / (2 * math.sqrt(self.config.hidden))
        unary = self.pair_unary(port)
        logits = logits + unary + unary.transpose(1, 2)
        width = port.shape[1]
        geom = np.zeros((len(bags), width, width, 2), dtype=np.float32)
        for i, g in enumerate(geometries):
            geom[i, :len(g), :len(g)] = g
        logits = logits + self.pair_geometry(torch.tensor(geom, device=self.device)).squeeze(-1)
        return [logits[i, :n, :n].float() for i, n in enumerate(lengths)]

    def equivalent_targets(self, ids, target, known, seed):
        variants = {tuple(tuple(e) for e in sorted(target))}
        layout = self.cat.layout(ids)
        offsets = layout["group_offsets"]
        rng = random.Random(seed)
        copies = {}
        for i, fid in enumerate(ids):
            copies.setdefault(fid, []).append(i)
        known_set = {tuple(sorted(e)) for e in known}
        for _ in range(self.config.symmetry_trials):
            destination = list(range(len(ids)))
            for positions in copies.values():
                shuffled = list(positions)
                rng.shuffle(shuffled)
                for a, b in zip(positions, shuffled):
                    destination[a] = b
            perm = []
            for i, fid in enumerate(ids):
                local = rng.choice(self.cat.symmetries[fid])
                perm.extend(offsets[destination[i]] + p for p in local)
            transformed_known = {tuple(sorted((perm[u], perm[v]))) for u, v in known}
            if transformed_known == known_set:
                variants.add(tuple(sorted(tuple(sorted((perm[u], perm[v]))) for u, v in target)))
        return [list(v) for v in variants]

    def connection_loss(self, bags, tasks, logits):
        losses, correct, count = [], 0., 0
        for ids, task, score in zip(bags, tasks, logits):
            remaining = torch.tensor(self.cat.remaining(ids, task['known']), device=self.device)
            active = remaining > 0
            if not active.any():
                continue
            mask = torch.tensor(self.cat.pair_mask(ids, task['known']), device=self.device)
            lp = F.log_softmax(score.masked_fill(~mask, -1e4), -1)
            known = {tuple(sorted(e)) for e in task['known']}
            options = []
            for variant in self.equivalent_targets(ids, task['target'], task['known'], task['seed']):
                truth = torch.zeros_like(score)
                for u, v in variant:
                    if (u, v) not in known:
                        truth[u, v] = truth[v, u] = 1
                if (truth.bool() & ~mask).any():
                    continue
                ce = -(truth[active]*lp[active]).sum(-1)/remaining[active]
                triangular = torch.triu(mask, diagonal=1)
                positive = triangular & truth.bool(); negative = triangular & ~truth.bool()
                bce = F.softplus(-score[positive]).mean()
                if negative.any():
                    bce = (bce + F.softplus(score[negative]).mean())/2
                options.append((ce.mean() + .25*bce, truth))
            loss, truth = min(options, key=lambda x: float(x[0].detach()))
            losses.append(loss)
            choice = score.masked_fill(~mask, -1e4).argmax(-1)
            correct += float(truth[active, choice[active]].sum().detach()); count += int(active.sum())
        zero = self.fragment_id.weight.sum()*0
        return (torch.stack(losses).mean() if losses else zero), correct, count

    def score_graphs(self, graphs, ctx):
        g, atoms, _ = self.chem(graphs, self.device)
        x, valid = padded_tokens(atoms, [len(a['x']) for a in graphs])
        with sdpa_kernel(SDPBackend.MATH):
            attention, _ = self.match_context(x, ctx['memory'], ctx['memory'],
                                             key_padding_mask=~ctx['mask'], need_weights=False)
        w = valid.to(x.dtype).unsqueeze(-1)
        matched = (attention*w).sum(1)/w.sum(1)
        z = ctx['z']
        return self.match_head(torch.cat((g, z, (g-z).abs(), g*z, matched), -1)).squeeze(-1).float()

    def ranking_loss(self, rows, ctx, negative_lists=None, in_batch=True, per_record=False,
                     negative_graphs=None):
        if negative_graphs is not None and not any(negative_graphs) and not in_batch:
            return ctx['z'].sum()*0
        true_graphs = [{'x': r['graph_x'], 'edges': r['graph_edges']} for r in rows]
        positives = self.score_graphs(true_graphs, ctx)
        graphs, owners = [], []
        for i, r in enumerate(rows):
            if negative_graphs is not None:
                graphs.extend(negative_graphs[i]); owners.extend([i]*len(negative_graphs[i]))
            else:
                for s in (r.get('negative_smiles', []) if negative_lists is None else negative_lists[i]):
                    graph = negative_graph(s)
                    if graph is not None:
                        graphs.append(graph); owners.append(i)
            if not in_batch:
                continue
            other = next((j for j, x in enumerate(rows) if x['mol_id'] != r['mol_id'] and x['formula'] == r['formula']), None)
            if other is None:
                other = next((j for j, x in enumerate(rows) if x['mol_id'] != r['mol_id']), None)
            if other is not None:
                graphs.append(true_graphs[other]); owners.append(i)
        if not graphs:
            return positives.sum()*0
        idx = torch.tensor(owners, device=self.device)
        negatives = self.score_graphs(graphs, {k: v[idx] for k, v in ctx.items()})
        pair_loss = F.softplus(.5 + negatives - positives[idx])
        if not per_record:
            return pair_loss.mean()
        total = pair_loss.new_zeros(len(rows)).index_add(0, idx, pair_loss)
        count = torch.bincount(idx, minlength=len(rows)).to(pair_loss.dtype)
        return (total[count > 0]/count[count > 0]).mean()

    def hard_ranking_loss(self, rows, inputs, negative_lists):
        """仅谱图条件与结构排序；不生成候选，不计算Add/Connect。"""
        ctx = self.encode_condition(inputs)
        return self.ranking_loss(rows, ctx, negative_lists, in_batch=False, per_record=True)

    def contrastive_alignment(self, rows, ctx, candidate_graphs):
        """每条谱图仅和自己的真图/困难图比较；冻结教师，谱编码器接收梯度。"""
        graphs, owners, lengths = [], [], []
        for i, negatives in enumerate(candidate_graphs):
            if not negatives:
                continue
            graphs.append({'x': rows[i]['graph_x'], 'edges': rows[i]['graph_edges']})
            graphs.extend(negatives)
            lengths.append(1 + len(negatives))
            owners.extend([i] * lengths[-1])
        if not graphs:
            return ctx['z'].sum()*0
        temperature = self.config.align_temperature
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('align_temperature必须为有限正数')
        teacher = F.normalize(self.graph_teacher(graphs, self.device), dim=-1)
        z = F.normalize(ctx['z'].float(), dim=-1)
        idx = torch.tensor(owners, device=self.device)
        scores = (z[idx] * teacher).sum(-1) / temperature
        logits = nn.utils.rnn.pad_sequence(scores.split(lengths), batch_first=True,
                                           padding_value=-torch.inf)
        return F.cross_entropy(logits, torch.zeros(len(lengths), dtype=torch.long, device=self.device))

    def auxiliary_loss(self, rows, inputs, ctx, keys=None, candidate_graphs=None,
                       tasks=None, remaining_logits=None):
        z = ctx['z']
        bank = self.keys() if keys is None else keys
        if tasks is None or remaining_logits is None:
            raise ValueError('剩余Presence需要当前Add任务和同一次前向的预测')
        logits = remaining_logits
        target = torch.zeros_like(logits)
        for i, task in enumerate(tasks):
            target[i, [f for f, n in task['remaining'].items() if n > 0]] = 1.
        element_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        # 每条记录内先平衡正、负例，避免几千个负例淹没少数真实片段。
        positive = (element_loss * target).sum(-1) / target.sum(-1).clamp_min(1)
        negative_mask = 1. - target
        negative = (element_loss * negative_mask).sum(-1) / negative_mask.sum(-1).clamp_min(1)
        # 完整集合没有正类：仍监督所有片段均不再需要。
        groups = (target.sum(-1) > 0).float() + (negative_mask.sum(-1) > 0).float()
        presence = ((positive + negative) / groups.clamp_min(1)).mean()
        counts = torch.tensor([len(r['fragment_ids']) for r in rows], device=self.device)
        if ((counts < 1) | (counts > self.config.max_heavy)).any():
            raise ValueError('真实片段数量超出1..max_heavy；请检查数据')
        count = F.cross_entropy(self.count_head(z).float(), counts)
        align = z.sum() * 0.
        if self.config.graph_alignment:
            align = self.contrastive_alignment(rows, ctx, candidate_graphs)
        return dict(presence=presence, count=count, align=align)

    def loss(self, rows, tasks, inputs, structure=False, candidate_graphs=None):
        use_candidates = self.config.auxiliary_tasks and not structure
        if use_candidates and (candidate_graphs is None or len(candidate_graphs) != len(rows)):
            raise ValueError('NMRexp损失需要当前batch逐记录的融合图候选')
        graphs = [{'x': r['graph_x'], 'edges': r['graph_edges']} for r in rows] if structure else None
        ctx = self.encode_condition(inputs, graphs)
        # Add/存在性/Connect共用本次编码；解冻时绝不跨参数更新缓存计算图。
        fragment_bank = self.fragment_bank() if self.config.fragment_repr in ('gnn', 'hybrid') else None
        keys = self.keys(fragment_bank=fragment_bank)
        scores, legal, remaining = self.add_logits(inputs, [t['prefix'] for t in tasks],
                                                  ctx, keys, return_presence=True)
        target = torch.zeros_like(scores, dtype=torch.float32)
        for i, t in enumerate(tasks):
            total = sum(t['remaining'].values())
            if total:
                for f, n in t['remaining'].items():
                    target[i, f] = n/total
            else:
                target[i, -1] = 1
        if (target.bool() & ~legal).any():
            raise ValueError('真实ADD目标不满足约束，请检查词表/数据')
        # 频次加权CE，非正例位置先清零以避免0*(-inf)。
        log_p = add_log_probs(scores, legal).masked_fill(~target.bool(), 0.)
        add = -(target*log_p).sum(-1).mean()
        bags = [r['fragment_ids'] for r in rows]
        connection = self.connection_logits(bags, [t['known'] for t in tasks], ctx, fragment_bank)
        connect, correct, count = self.connection_loss(bags, tasks, connection)
        if self.config.rank_weight and use_candidates:
            rank = self.ranking_loss(rows, ctx, in_batch=False, per_record=True,
                                     negative_graphs=candidate_graphs)
        else:
            rank = self.ranking_loss(rows, ctx) if self.config.rank_weight else add*0
        total = add + self.config.connect_weight*connect + self.config.rank_weight*rank
        losses = {'total': total, 'add': add, 'connect': connect, 'rank': rank}
        if self.config.auxiliary_tasks and not structure:
            aux = self.auxiliary_loss(rows, inputs, ctx, keys, candidate_graphs, tasks, remaining)
            losses.update(aux)
            losses['total'] = total + sum(NMR_AUX_WEIGHTS[k] * aux[k] for k in aux)
        return losses

    def checkpoint_config(self):
        return asdict(self.config)
