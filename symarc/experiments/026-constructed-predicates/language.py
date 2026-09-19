"""Finite typed expression grammar. No named shape predicates or learned atoms.

Aliases expand before costing or deduplication. Only exact commutativity is used
for canonicalization; agreement on observations is never a rewrite rule.
"""
from __future__ import annotations
from functools import lru_cache
import json
from typing import Iterable

Expr = tuple
ATOMS = (("count",), ("span_r",), ("span_c",), ("lit", 0), ("lit", 1))
COMMUTATIVE = frozenset(("add", "mul", "eq", "and"))


def code(e: Expr) -> str:
    return json.dumps(e, separators=(",", ":"))


def canon(e, aliases: dict | None = None, active: frozenset = frozenset()) -> Expr:
    aliases = aliases or {}
    if not isinstance(e, (tuple, list)) or not e:
        raise ValueError("An expression must be a nonempty tuple/list")
    op = e[0]
    if op == "alias":
        if len(e) != 2 or e[1] not in aliases or e[1] in active:
            raise ValueError("Missing or cyclic alias")
        return canon(aliases[e[1]], aliases, active | {e[1]})
    if op in ("count", "span_r", "span_c"):
        if len(e) != 1: raise ValueError("Invalid scalar observation")
        return (op,)
    if op == "lit":
        if len(e) != 2 or type(e[1]) is not int or e[1] not in (0, 1):
            raise ValueError("Only zero and one are grammar literals")
        return (op, e[1])
    arity = 1 if op == "not" else 2
    if op not in ("add", "sub", "mul", "eq", "lt", "not", "and") or len(e) != arity + 1:
        raise ValueError("Unknown constructor or wrong arity")
    args = tuple(canon(x, aliases, active) for x in e[1:])
    required = "B" if op in ("not", "and") else "I"
    if any(sort(x) != required for x in args): raise ValueError("Ill-typed expression")
    if op in COMMUTATIVE: args = tuple(sorted(args, key=code))
    return (op, *args)


def sort(e: Expr) -> str:
    return "B" if e[0] in ("eq", "lt", "not", "and") else "I"


def cost(e: Expr) -> int:
    if e[0] in ("count", "span_r", "span_c", "lit"): return 1
    if e[0] == "alias": raise ValueError("Cost requires expanded syntax")
    return 1 + sum(cost(x) for x in e[1:])


def observations(points: Iterable) -> tuple[int, int, int]:
    ps = set()
    for p in points:
        if len(p) != 2 or any(type(v) is not int for v in p):
            raise ValueError("Coordinates must be integer pairs")
        ps.add(tuple(p))
    if not ps: raise ValueError("Empty sets are outside this experiment")
    rs = [r for r, c in ps]; cs = [c for r, c in ps]
    return len(ps), max(rs)-min(rs)+1, max(cs)-min(cs)+1


def evaluate(e: Expr, x: tuple[int, int, int]):
    op = e[0]
    if op == "count": return x[0]
    if op == "span_r": return x[1]
    if op == "span_c": return x[2]
    if op == "lit": return e[1]
    a = evaluate(e[1], x)
    if op == "not": return not a
    b = evaluate(e[2], x)
    if op == "add": return a+b
    if op == "sub": return a-b
    if op == "mul": return a*b
    if op == "eq": return a == b
    if op == "lt": return a < b
    if op == "and": return a and b
    raise ValueError(op)


def grammar(bound: int = 5, aliases: dict | None = None) -> list[Expr]:
    if bound < 1: return []
    pools = {s: {k: set() for k in range(1, bound+1)} for s in ("I", "B")}
    for e in ATOMS: pools["I"][1].add(e)
    for name in aliases or {}:
        e = canon(("alias", name), aliases)
        if cost(e) <= bound: pools[sort(e)][cost(e)].add(e)
    for k in range(2, bound+1):
        for b in list(pools["B"][k-1]): pools["B"][k].add(("not", b))
        for i in range(1, k-1):
            j = k-1-i
            for typ, ops in (("I", ("add", "sub", "mul", "eq", "lt")), ("B", ("and",))):
                for a in list(pools[typ][i]):
                    for b in list(pools[typ][j]):
                        for op in ops:
                            e = canon((op, a, b))
                            pools[sort(e)][k].add(e)
    return sorted((e for s in pools["B"].values() for e in s), key=lambda e: (cost(e), code(e)))


def render(e: Expr) -> str:
    leaves = {"count": "count(S)", "span_r": "span(rows(S))", "span_c": "span(cols(S))"}
    if e[0] in leaves: return leaves[e[0]]
    if e[0] == "lit": return str(e[1])
    if e[0] == "not": return "not " + render(e[1])
    symbols = {"add": "+", "sub": "-", "mul": "*", "eq": "=", "lt": "<", "and": "and"}
    return f"({render(e[1])} {symbols[e[0]]} {render(e[2])})"
