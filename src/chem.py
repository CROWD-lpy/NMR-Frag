"""化学约定：沿用03词表和显式氢；合并同宿主、同键阶的重复端口。"""
from __future__ import annotations
import hashlib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

ELEMENTS = ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se")
HEAVY = tuple(x for x in ELEMENTS if x != "H")
ATOM_DIM = len(HEAVY) + 9
BONDS = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
VERSION = "nmr_fragment_v3"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def formula_vector(text):
    terms = re.findall(r"([A-Z][a-z]?)(\d*)", text or "")
    if "".join(e + n for e, n in terms) != text or any(e not in ELEMENTS for e, _ in terms):
        raise ValueError(f"不支持的分子式: {text}")
    counts = Counter()
    for e, n in terms:
        counts[e] += int(n or 1)
    return np.asarray([counts[e] for e in ELEMENTS], dtype=np.int64)


def canonical(smiles):
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无效SMILES: {smiles}")
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def graph_data(mol):
    """保留每个真实原子；双向边由模型打包时产生。"""
    mol.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(mol)  # 只使用是否在环中的标记，无需枚举最小环集合。
    real = [a for a in mol.GetAtoms() if a.GetAtomicNum()]
    remap = {a.GetIdx(): i for i, a in enumerate(real)}
    x = []
    for a in real:
        if a.GetSymbol() not in HEAVY:
            raise ValueError(f"不支持元素: {a.GetSymbol()}")
        bonds = [b for b in a.GetBonds() if b.GetOtherAtom(a).GetAtomicNum()]
        orders = [4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble()) for b in bonds]
        x.append([float(a.GetSymbol() == e) for e in HEAVY] + [
            a.GetTotalNumHs() / 4, len(bonds) / 4,
            (a.GetDegree() - len(bonds)) / 4, float(a.GetIsAromatic()),
            float(a.IsInRing()), sum(b.GetBondTypeAsDouble() for b in a.GetBonds()) / 6,
            orders.count(1) / 4, orders.count(2) / 4, orders.count(3) / 4])
    edges = []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if u in remap and v in remap:
            edges.append([remap[u], remap[v], 4 if b.GetIsAromatic() else int(b.GetBondTypeAsDouble())])
    return {"x": x, "edges": edges}


@lru_cache(maxsize=1024)
def negative_graph(smiles):
    """缓存可用的负样本图；解析或特征提取失败时返回 None。"""
    with rdBase.BlockLogs():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or not mol.GetNumHeavyAtoms():
                return None
            return graph_data(mol)
        except (ValueError, RuntimeError):
            return None


class Catalog:
    def __init__(self, path):
        path = Path(path)
        if path.suffix == ".pt":
            data = torch.load(path, map_location="cpu", weights_only=False)
            if data["version"] != VERSION:
                raise ValueError("04 catalog版本不兼容")
            self.__dict__.update(data)
            self.formulas = np.asarray(self.formulas, dtype=np.int64)
            return
        self.tokens = json.loads(path.read_text(encoding="utf-8"))["tokens"]
        self.tokenizer_sha256 = digest(path)
        self.version = VERSION
        self.graphs, self.groups, self.port_to_group, self.symmetries = [], [], [], []
        self.formulas, self.unsaturation = [], []
        with rdBase.BlockLogs():
            for token in self.tokens:
                raw = Chem.MolFromSmiles(token, sanitize=False)
                if raw is None:
                    raise ValueError(f"无效片段: {token}")
                raw.UpdatePropertyCache(strict=False)
                real = [a for a in raw.GetAtoms() if a.GetAtomicNum()]
                remap = {a.GetIdx(): i for i, a in enumerate(real)}
                ports = []
                for a in raw.GetAtoms():
                    if a.GetAtomicNum() == 0:
                        if a.GetDegree() != 1:
                            raise ValueError(f"端口必须只有一个宿主: {token}")
                        host = a.GetNeighbors()[0].GetIdx()
                        order = int(raw.GetBondBetweenAtoms(a.GetIdx(), host).GetBondTypeAsDouble())
                        if order not in BONDS:
                            raise ValueError("03切口应是凯库勒单/双/三键")
                        ports.append((remap[host], order))
                counts = Counter(ports)
                keys = sorted(counts)
                groups = [{"host": h, "order": o, "capacity": counts[h, o]} for h, o in keys]
                mapping = {key: i for i, key in enumerate(keys)}
                self.groups.append(groups)
                self.port_to_group.append([mapping[p] for p in ports])
                f = Counter(a.GetSymbol() for a in real)
                f["H"] = sum(a.GetTotalNumHs() for a in real)
                self.formulas.append([f[e] for e in ELEMENTS])
                internal = [b for b in raw.GetBonds() if b.GetBeginAtom().GetAtomicNum() and b.GetEndAtom().GetAtomicNum()]
                rings = max(0, len(internal) - len(real) + 1)
                self.unsaturation.append(rings + sum(b.GetBondTypeAsDouble() - 1 for b in internal) +
                                         sum(o - 1 for _, o in ports) / 2)
                mol = Chem.Mol(raw)
                try:
                    Chem.SanitizeMol(mol)
                except (ValueError, RuntimeError):
                    mol = raw
                self.graphs.append(graph_data(mol))
                # 只保存少量整体自同构，绝不把不同宿主永久合并。
                sym = {tuple(range(len(groups)))}
                if len(groups) > 1:
                    for match in mol.GetSubstructMatches(mol, uniquify=False, maxMatches=16):
                        perm = []
                        for h, o in keys:
                            original = real[h].GetIdx()
                            mapped = remap.get(match[original])
                            key = (mapped, o)
                            if key not in mapping or counts[key] != counts[h, o]:
                                break
                            perm.append(mapping[key])
                        if len(perm) == len(groups):
                            sym.add(tuple(perm))
                self.symmetries.append([list(p) for p in sorted(sym)])
        self.formulas = np.asarray(self.formulas, dtype=np.int64)
        self.port_totals = np.asarray([sum(g["capacity"] for g in gs) for gs in self.groups])
        self.max_h_per_element = np.zeros(len(HEAVY), dtype=np.float32)
        for graph, ports in zip(self.graphs, self.port_totals):
            if ports:
                for x in graph["x"]:
                    e = int(np.argmax(x[:len(HEAVY)]))
                    self.max_h_per_element[e] = max(self.max_h_per_element[e], round(x[len(HEAVY)] * 4))

    def __len__(self):
        return len(self.tokens)

    def save(self, path):
        data = dict(self.__dict__)
        data["formulas"] = self.formulas.tolist()
        torch.save(data, path)

    def layout(self, ids):
        return self._layout(tuple(ids))

    @lru_cache(maxsize=1024)
    def _layout(self, ids):
        groups, atom_offset, group_offsets, atom_offsets, internal = [], 0, [], [], []
        for instance, fid in enumerate(ids):
            group_offsets.append(len(groups))
            atom_offsets.append(atom_offset)
            for g in self.groups[fid]:
                groups.append({**g, "instance": instance, "atom": atom_offset + g["host"], "fid": fid})
            internal.extend([[atom_offset + u, atom_offset + v, o] for u, v, o in self.graphs[fid]["edges"]])
            atom_offset += len(self.graphs[fid]["x"])
        return {"groups": groups, "group_offsets": group_offsets, "atom_offsets": atom_offsets,
                "internal": internal, "n_atoms": atom_offset,
                "owners": np.asarray([g["instance"] for g in groups]),
                "orders": np.asarray([g["order"] for g in groups]),
                "atoms": np.asarray([g["atom"] for g in groups]),
                "capacities": np.asarray([g["capacity"] for g in groups], dtype=np.int64)}

    def target(self, actions):
        ids = [int(a["fragment_id"]) for a in actions if a["type"] == 0]
        if not ids or any(f < 0 or f >= len(self) for f in ids):
            raise ValueError("ADD片段ID不在03词表中")
        layout = self.layout(ids)
        edges, used = [], set()
        for a in actions:
            if a["type"] != 1:
                continue
            ends = []
            for side in ("left", "right"):
                instance, port = int(a[side]), int(a[side + "_port"])
                if (instance, port) in used:
                    raise ValueError("03目标重复使用原始端口")
                used.add((instance, port))
                local = self.port_to_group[ids[instance]][port]
                ends.append(layout["group_offsets"][instance] + local)
            u, v = sorted(ends)
            if layout["groups"][u]["order"] != int(a["order"]):
                raise ValueError("03目标键阶不匹配")
            edges.append([u, v])
        return ids, sorted(edges)

    def remaining(self, ids, edges):
        r = self.layout(ids)["capacities"].copy()
        for u, v in edges:
            r[u] -= 1
            r[v] -= 1
        return r

    def pair_mask(self, ids, edges):
        layout = self.layout(ids)
        groups = layout["groups"]
        n = len(groups)
        if not n:
            return np.zeros((0, 0), dtype=bool)
        r = self.remaining(ids, edges)
        inst, order = layout["owners"], layout["orders"]
        mask = (r[:, None] > 0) & (r[None, :] > 0) & (inst[:, None] != inst[None, :])
        mask &= order[:, None] == order[None, :]
        atoms = layout["atoms"]
        for u, v in edges:
            a = np.flatnonzero(atoms == groups[u]["atom"])
            b = np.flatnonzero(atoms == groups[v]["atom"])
            mask[np.ix_(a, b)] = False
            mask[np.ix_(b, a)] = False
        return mask

    def can_finish_bag(self, ids):
        if not ids:
            return False
        if len(ids) == 1 and self.port_totals[ids[0]] == 0:
            return True
        if any(self.port_totals[f] == 0 for f in ids):
            return False
        groups = self.layout(ids)["groups"]
        by_order = Counter()
        for g in groups:
            by_order[g["order"]] += g["capacity"]
        if any(k % 2 for k in by_order.values()) or sum(by_order.values()) // 2 < len(ids) - 1:
            return False
        return bool(np.all(self.pair_mask(ids, []).sum(1) >= self.remaining(ids, [])))

    def partial_feasible(self, ids, edges):
        r = self.remaining(ids, edges)
        if np.any(r < 0) or np.any(self.pair_mask(ids, edges).sum(1) < r):
            return False
        # 不允许在其他片段尚未接入时，把一个连通分量的所有切口封死。
        parents = list(range(len(ids)))
        def root(i):
            while parents[i] != i:
                i = parents[i]
            return i
        groups = self.layout(ids)["groups"]
        for u, v in edges:
            parents[root(groups[u]["instance"])] = root(groups[v]["instance"])
        comps = {root(i) for i in range(len(ids))}
        if len(comps) > 1:
            open_comps = {root(g["instance"]) for g, k in zip(groups, r) if k > 0}
            if open_comps != comps:
                return False
        return True

    def rebuild(self, ids, edges):
        layout = self.layout(ids)
        groups = layout["groups"]
        if not ids or np.any(self.remaining(ids, edges)) or not self.partial_feasible(ids, edges):
            return None
        try:
            with rdBase.BlockLogs():
                rw = Chem.RWMol()
                for fid in ids:
                    mol = Chem.MolFromSmiles(self.tokens[fid], sanitize=False)
                    remap = {}
                    for a in mol.GetAtoms():
                        if a.GetAtomicNum():
                            remap[a.GetIdx()] = rw.AddAtom(Chem.Atom(a))
                    for b in mol.GetBonds():
                        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                        if u in remap and v in remap:
                            rw.AddBond(remap[u], remap[v], b.GetBondType())
                seen = set()
                for u, v in edges:
                    a, b = groups[u], groups[v]
                    key = tuple(sorted((a["atom"], b["atom"])))
                    if a["instance"] == b["instance"] or a["order"] != b["order"] or key in seen:
                        return None
                    seen.add(key)
                    rw.AddBond(a["atom"], b["atom"], BONDS[a["order"]])
                mol = rw.GetMol()
                Chem.SanitizeMol(mol)
                if len(Chem.GetMolFrags(mol)) != 1 or any(a.GetFormalCharge() or a.GetNumRadicalElectrons()
                                                         or a.GetIsotope() for a in mol.GetAtoms()):
                    return None
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
                formula = rdMolDescriptors.CalcMolFormula(mol)
                parsed = Chem.MolFromSmiles(smiles)
                if parsed is None or rdMolDescriptors.CalcMolFormula(parsed) != formula:
                    return None
                return smiles, formula
        except (ValueError, RuntimeError, IndexError):
            return None
