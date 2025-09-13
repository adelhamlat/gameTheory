
from __future__ import annotations
from typing import List, Dict
import sympy as sp

def gradient(expr: sp.Expr, vars_syms: List[sp.Symbol]) -> sp.Matrix:
    return sp.Matrix([sp.diff(expr, v) for v in vars_syms])

def second_derivatives(expr: sp.Expr, vars_syms: List[sp.Symbol]) -> Dict[sp.Symbol, sp.Expr]:
    return {v: sp.simplify(sp.diff(expr, v, 2)) for v in vars_syms}
