
from __future__ import annotations
from typing import Dict, Optional, Sequence
import sympy as sp

def try_solve_symbolic(eqs: Sequence[sp.Eq], vars_syms: Sequence[sp.Symbol]) -> Optional[Dict[sp.Symbol, sp.Expr]]:
    try:
        sol = sp.solve(list(eqs), list(vars_syms), dict=True)
        if isinstance(sol, list) and sol:
            return {k: sp.simplify(v) for k,v in sol[len(sol)-1].items()}
    except Exception:
        pass
    try:
        solset = sp.nonlinsolve(eqs, vars_syms)
        if solset:
            for tup in solset:
                return {var: sp.simplify(val) for var, val in zip(vars_syms, tup)}
    except Exception:
        pass
    return None
