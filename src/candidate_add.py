"""两阶段共享：保留条件/集合读取，恢复余弦主分数与证据点积修正。"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel


class CandidateConditionAttention(nn.Module):
    """PubChem读取图条件，NMRexp读取谱条件；无候选间自注意力。"""
    def __init__(self, config):
        super().__init__()
        h, d = config.hidden, config.candidate_hidden
        self.query = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, d))
        self.memory_norm = nn.LayerNorm(h)
        self.attention = nn.ModuleList(nn.MultiheadAttention(
            d, config.candidate_heads, config.dropout, kdim=h, vdim=h, batch_first=True)
            for _ in range(config.candidate_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in self.attention)

    @sdpa_kernel(SDPBackend.MATH)
    def forward(self, bank, memory, valid):
        query = self.query(bank)[None].expand(len(memory), -1, -1)
        memory = self.memory_norm(memory)
        for attention, norm in zip(self.attention, self.norms):
            readout, _ = attention(query, memory, memory,
                key_padding_mask=~valid, need_weights=False)
            query = norm(query + readout)
        return query


class CandidateAddHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        h, d = config.hidden, config.candidate_hidden
        # 三项相加等价于一个Linear(concat(e_f, E_f, q_t))。
        self.query_fragment = nn.Linear(h, d, bias=False)
        self.query_evidence = nn.Linear(d, d, bias=False)
        self.query_state = nn.Linear(h, d)
        self.query_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(h)
        self.attention = nn.ModuleList(nn.MultiheadAttention(
            d, config.candidate_heads, config.dropout, kdim=h, vdim=h, batch_first=True)
            for _ in range(config.candidate_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in self.attention)
        # v5评分形式：归一化状态与片段余弦 + 条件证据点积。
        # 证据改用集合读取后的C，保留v6已有的组合信息路径。
        self.logit_scale = nn.Parameter(torch.tensor(math.log(16.)))
        self.candidate_state = nn.Linear(h, d)
        nn.init.zeros_(self.candidate_state.weight)
        nn.init.zeros_(self.candidate_state.bias)

    @sdpa_kernel(SDPBackend.MATH)
    def forward(self, bank, evidence, state, selected, valid, global_condition):
        # bank:[V,h]；evidence:[B或1,V,d]，生成时同一条件证据广播到beam。
        query = self.query_norm(self.query_fragment(bank)[None]
            + self.query_evidence(evidence) + self.query_state(state)[:, None])
        # selected含START及已选片段；显式增加全局条件z，空prefix也不会全mask。
        memory = self.memory_norm(torch.cat((global_condition[:, None], selected), 1))
        mask = torch.cat((torch.ones((len(valid), 1), dtype=torch.bool, device=valid.device), valid), 1)
        for attention, norm in zip(self.attention, self.norms):
            readout, _ = attention(query, memory, memory,
                key_padding_mask=~mask, need_weights=False)
            query = norm(query + readout)
        # bank由Model.keys归一化；温度初值16、上限100，与v5一致。
        cosine = F.normalize(state, dim=-1) @ bank.T
        scale = self.logit_scale.clamp(max=math.log(100.)).exp()
        matched = torch.matmul(query, self.candidate_state(state).unsqueeze(-1)).squeeze(-1)
        return cosine.float()*scale + matched.float()/math.sqrt(query.shape[-1])
