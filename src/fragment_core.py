"""Bounded frequent-merge fragmenter; output is a port graph, not an action sequence.

Public API: Fragmenter.fit(sample_smiles), .collect_vocab(train_smiles),
.freeze(), .decompose(smiles), .save(path), .load(path), rebuild(partition, vocab).
"""
from __future__ import annotations

import heapq
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from data_io import normalize_smiles, write_json


BT = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}


@dataclass
class Node:
    atoms: frozenset
    children: tuple = ()


class MoleculeGraph:
    def __init__(self, smiles):
        info = normalize_smiles(smiles)
        self.info = info
        self.mol = Chem.MolFromSmiles(info["smiles"])
        hydrogens = [a.GetTotalNumHs() for a in self.mol.GetAtoms()]
        Chem.Kekulize(self.mol, clearAromaticFlags=True)
        for atom, h in zip(self.mol.GetAtoms(), hydrogens):
            atom.SetNumExplicitHs(h)
            atom.SetNoImplicit(True)
        self.mol.UpdatePropertyCache(strict=True)
        self.bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble()))
                      for b in self.mol.GetBonds()]
        self.pattern_cache, self.token_cache = {}, {}

    def pattern(self, atoms):
        atoms = frozenset(atoms)
        if atoms not in self.pattern_cache:
            rw, remap = Chem.RWMol(), {}
            for i in sorted(atoms):
                a = Chem.Atom(self.mol.GetAtomWithIdx(i).GetAtomicNum())
                a.SetNoImplicit(True)
                remap[i] = rw.AddAtom(a)
            for a, b, order in self.bonds:
                if a in atoms and b in atoms:
                    rw.AddBond(remap[a], remap[b], BT[order])
            mol = rw.GetMol()
            mol.UpdatePropertyCache(strict=False)
            self.pattern_cache[atoms] = Chem.MolToSmiles(
                mol, canonical=True, isomericSmiles=False, kekuleSmiles=True)
        return self.pattern_cache[atoms]

    def port_token(self, atoms):
        """Return (canonical token, original bond index -> local dummy/port index)."""
        atoms = frozenset(atoms)
        if atoms not in self.token_cache:
            rw, remap, cuts = Chem.RWMol(), {}, []
            for i in sorted(atoms):
                remap[i] = rw.AddAtom(Chem.Atom(self.mol.GetAtomWithIdx(i)))
            for bi, (a, b, order) in enumerate(self.bonds):
                if a in atoms and b in atoms:
                    rw.AddBond(remap[a], remap[b], BT[order])
                elif (a in atoms) != (b in atoms):
                    inside = a if a in atoms else b
                    dummy = Chem.Atom(0)
                    dummy.SetNoImplicit(True)
                    di = rw.AddAtom(dummy)
                    rw.AddBond(remap[inside], di, BT[order])
                    cuts.append((bi, di))
            mol = rw.GetMol()
            mol.UpdatePropertyCache(strict=True)
            token = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False,
                                     kekuleSmiles=True, allHsExplicit=True)
            atom_order = json.loads(mol.GetProp("_smilesAtomOutputOrder"))
            position = {atom: pos for pos, atom in enumerate(atom_order)}
            cuts.sort(key=lambda cut: position[cut[1]])
            self.token_cache[atoms] = token, {bi: pi for pi, (bi, _) in enumerate(cuts)}
        return self.token_cache[atoms]


class MergeState:
    def __init__(self, graph):
        self.graph = graph
        n = graph.mol.GetNumAtoms()
        self.nodes = {i: Node(frozenset((i,))) for i in range(n)}
        self.owner = list(range(n))
        self.next_id = n

    def pairs(self):
        return sorted({tuple(sorted((self.owner[a], self.owner[b])))
                       for a, b, _ in self.graph.bonds if self.owner[a] != self.owner[b]})

    def candidates(self, max_atoms):
        result = []
        for a, b in self.pairs():
            union = self.nodes[a].atoms | self.nodes[b].atoms
            if len(union) <= max_atoms:
                result.append((self.graph.pattern(union), a, b))
        return result

    def merge(self, a, b):
        left, right = self.nodes.pop(a), self.nodes.pop(b)
        new_id = self.next_id
        self.next_id += 1
        atoms = left.atoms | right.atoms
        self.nodes[new_id] = Node(atoms, (left, right))
        for atom in atoms:
            self.owner[atom] = new_id
        return new_id

    def apply(self, pattern, max_atoms):
        hits = [(a, b) for key, a, b in self.candidates(max_atoms) if key == pattern]
        hits.sort(key=lambda p: (tuple(sorted(self.nodes[p[0]].atoms)),
                                 tuple(sorted(self.nodes[p[1]].atoms))))
        count = 0
        for a, b in hits:
            if a in self.nodes and b in self.nodes:
                self.merge(a, b)
                count += 1
        return count

    def roots(self):
        return sorted(self.nodes.values(), key=lambda node: min(node.atoms))


def atomic_tokens():
    """Finite neutral atom/port templates; no eval structures are used here."""
    valences = {"C": (4,), "N": (3,), "O": (2,), "P": (3, 5),
                "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,), "B": (3,), "Si": (4,), "Se": (2, 4, 6)}
    tokens = set()
    with rdBase.BlockLogs():
        for element, allowed in valences.items():
            for valence in allowed:
                for degree in range(valence + 1):
                    for orders in itertools.combinations_with_replacement((1, 2, 3), degree):
                        h = valence - sum(orders)
                        if h < 0:
                            continue
                        rw = Chem.RWMol()
                        atom = Chem.Atom(element)
                        atom.SetNumExplicitHs(h)
                        atom.SetNoImplicit(True)
                        rw.AddAtom(atom)
                        for bo in orders:
                            dummy = Chem.Atom(0)
                            dummy.SetNoImplicit(True)
                            di = rw.AddAtom(dummy)
                            rw.AddBond(0, di, BT[bo])
                        mol = rw.GetMol()
                        try:
                            Chem.SanitizeMol(mol)
                            if mol.GetAtomWithIdx(0).GetNumRadicalElectrons() == 0:
                                tokens.add(Chem.MolToSmiles(
                                    mol, canonical=True, isomericSmiles=False,
                                    kekuleSmiles=True, allHsExplicit=True))
                        except (ValueError, RuntimeError):
                            continue
    return tokens


class Fragmenter:
    def __init__(self, max_atoms=12, n_merges=256):
        self.max_atoms, self.n_merges = max_atoms, n_merges
        self.rules, self.logs, self.known_tokens = [], [], set()
        self.vocab, self.id_to_token = {}, []
        self.mining_molecules = self.vocab_molecules = 0

    def fit(self, sample_smiles):
        """Learn at most K global patterns from distinct sampled structures."""
        states = [MergeState(MoleculeGraph(s)) for s in dict.fromkeys(sample_smiles)]
        self.mining_molecules = len(states)
        counts, locations, local = Counter(), defaultdict(set), []
        for i, state in enumerate(states):
            c = Counter(pattern for pattern, _, _ in state.candidates(self.max_atoms))
            local.append(c)
            counts.update(c)
            for pattern in c:
                locations[pattern].add(i)
        heap = [(-count, pattern) for pattern, count in counts.items()]
        heapq.heapify(heap)
        remaining = sum(len(state.nodes) for state in states)
        for step in range(self.n_merges):
            while heap:
                neg, pattern = heapq.heappop(heap)
                if counts[pattern] == -neg and -neg > 0:
                    break
            else:
                break
            frequency, actual, changed = counts[pattern], 0, set()
            for i in sorted(locations[pattern]):
                for key, num in local[i].items():
                    counts[key] -= num
                    locations[key].discard(i)
                    changed.add(key)
                actual += states[i].apply(pattern, self.max_atoms)
                local[i] = Counter(key for key, _, _ in states[i].candidates(self.max_atoms))
                for key, num in local[i].items():
                    counts[key] += num
                    locations[key].add(i)
                    changed.add(key)
            for key in changed:
                if counts[key] > 0:
                    heapq.heappush(heap, (-counts[key], key))
            remaining -= actual
            self.rules.append(pattern)
            self.logs.append({"step": step + 1, "frequency": frequency,
                              "merges": actual, "remaining": remaining})
        # Retain the sample's merge history as intermediate backoff vocabulary.
        def visit(node, graph):
            self.known_tokens.add(graph.port_token(node.atoms)[0])
            for child in node.children:
                visit(child, graph)
        for state in states:
            for node in state.roots():
                visit(node, state.graph)
        return self

    def replay(self, smiles):
        """Apply only rules that occur in this molecule, in exactly learned order."""
        state = MergeState(MoleculeGraph(smiles))
        if len(self.rules) != len(set(self.rules)):
            for pattern in self.rules:  # Defensive path for repeated patterns.
                state.apply(pattern, self.max_atoms)
            return state
        rank = {pattern: i for i, pattern in enumerate(self.rules)}
        heap = []

        def push(a, b, after):
            union = state.nodes[a].atoms | state.nodes[b].atoms
            if len(union) > self.max_atoms:
                return
            index = rank.get(state.graph.pattern(union))
            if index is not None and index > after:
                left = tuple(sorted(state.nodes[a].atoms))
                right = tuple(sorted(state.nodes[b].atoms))
                heapq.heappush(heap, (index, left, right, a, b))

        for a, b in state.pairs():
            push(a, b, -1)
        while heap:
            index, _, _, a, b = heapq.heappop(heap)
            if a not in state.nodes or b not in state.nodes:
                continue
            new_id = state.merge(a, b)
            neighbors = set()
            for u, v, _ in state.graph.bonds:
                if state.owner[u] == new_id and state.owner[v] != new_id:
                    neighbors.add(state.owner[v])
                if state.owner[v] == new_id and state.owner[u] != new_id:
                    neighbors.add(state.owner[u])
            for neighbor in neighbors:
                push(*sorted((new_id, neighbor)), after=index)
        return state

    def collect_vocab(self, train_smiles):
        """Replay final motifs across *all eligible train structures*, once."""
        for smiles in train_smiles:
            state = self.replay(smiles)
            self.known_tokens.update(state.graph.port_token(n.atoms)[0] for n in state.roots())
            self.vocab_molecules += 1
            if self.vocab_molecules % 100000 == 0:
                print(f"词表收集：{self.vocab_molecules:,} 个训练结构", flush=True)
        return self

    def freeze(self):
        self.id_to_token = sorted(self.known_tokens | atomic_tokens())
        self.vocab = {token: index for index, token in enumerate(self.id_to_token)}
        self.known_tokens.clear()
        return self

    def decompose(self, smiles):
        """Frozen vocabulary -> disjoint fragments and explicit cut-bond port pairs.

        No Add/Connect order is chosen here. Unknown final tokens are recursively
        split along the learned merge tree; unknown atomic tokens remain null IDs.
        """
        state = self.replay(smiles)
        roots = state.roots()
        raw_covered = all(state.graph.port_token(n.atoms)[0] in self.vocab for n in roots)
        leaves, backoffs = [], 0

        def append(node):
            nonlocal backoffs
            if state.graph.port_token(node.atoms)[0] in self.vocab or not node.children:
                leaves.append(node)
            else:
                backoffs += 1
                for child in node.children:
                    append(child)

        for node in roots:
            append(node)
        leaves.sort(key=lambda n: min(n.atoms))
        fragments, ends = [], defaultdict(list)
        for instance, node in enumerate(leaves):
            token, binding = state.graph.port_token(node.atoms)
            atoms = sorted(node.atoms)
            fragments.append({"fragment_id": self.vocab.get(token), "token": token,
                              "atoms": atoms, "heavy_atoms": len(atoms),
                              "ports": len(binding),
                              "is_carbon_singleton": len(atoms) == 1 and
                              state.graph.mol.GetAtomWithIdx(atoms[0]).GetAtomicNum() == 6})
            for bi, port in binding.items():
                ends[bi].append((instance, port))
        connections = []
        for bi in sorted(ends):
            (left, lp), (right, rp) = ends[bi]
            connections.append({"left": left, "left_port": lp, "right": right,
                                "right_port": rp, "order": state.graph.bonds[bi][2]})
        return {"smiles": state.graph.info["smiles"],
                "mol_id": state.graph.info["mol_id"], "formula": state.graph.info["formula"],
                "heavy_atoms": state.graph.info["heavy_atoms"],
                "fragments": fragments, "connections": connections,
                "raw_covered": raw_covered,
                "covered": all(f["fragment_id"] is not None for f in fragments),
                "backoff_splits": backoffs}

    def save(self, path):
        write_json(path, {"version": 1, "rdkit_version": rdBase.rdkitVersion,
                          "max_atoms": self.max_atoms, "n_merges": self.n_merges,
                          "rules": self.rules, "tokens": self.id_to_token,
                          "mining_molecules": self.mining_molecules,
                          "vocab_molecules": self.vocab_molecules})

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(value["max_atoms"], value["n_merges"])
        model.rules, model.id_to_token = value["rules"], value["tokens"]
        model.vocab = {token: i for i, token in enumerate(model.id_to_token)}
        model.mining_molecules = value["mining_molecules"]
        model.vocab_molecules = value["vocab_molecules"]
        return model


def rebuild(partition, id_to_token):
    """Independent 2D reconstruction from frozen token IDs and port connections."""
    rw = Chem.RWMol()
    ports, used = {}, set()
    for instance, fragment in enumerate(partition["fragments"]):
        mol = Chem.MolFromSmiles(id_to_token[fragment["fragment_id"]], sanitize=False)
        mol.UpdatePropertyCache(strict=True)
        remap, dummies = {}, []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum():
                remap[atom.GetIdx()] = rw.AddAtom(Chem.Atom(atom))
            else:
                dummies.append(atom.GetIdx())
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in remap and b in remap:
                rw.AddBond(remap[a], remap[b], bond.GetBondType())
        for port_id, dummy in enumerate(dummies):
            neighbor = mol.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
            order = int(mol.GetBondBetweenAtoms(dummy, neighbor).GetBondTypeAsDouble())
            ports[(instance, port_id)] = (remap[neighbor], order)
    for edge in partition["connections"]:
        left = (edge["left"], edge["left_port"])
        right = (edge["right"], edge["right_port"])
        a, order = ports[left]
        b, other = ports[right]
        if left in used or right in used or order != other or order != edge["order"]:
            raise ValueError("端口使用或键级错误")
        rw.AddBond(a, b, BT[order])
        used.update((left, right))
    if len(used) != len(ports):
        raise ValueError("仍有未连接端口")
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("重构结果不连通")
    return (Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
            rdMolDescriptors.CalcMolFormula(mol))
