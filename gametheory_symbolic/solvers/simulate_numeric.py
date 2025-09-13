
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import sympy as sp
from ..core.types import GameModel, Stage
from .simultaneous_symbolic import build_foc_equations_symbolic

def numeric_solve_given_params(game: GameModel, stage: Stage, param_values: Dict[str, float], x0: Optional[List[float]]=None):
    eqs, vars_syms = build_foc_equations_symbolic(game, stage)
    subs_map = {game.param_symbols[k]: float(v) for k, v in param_values.items()}
    eqs_num = [sp.simplify(e.lhs.subs(subs_map)) for e in eqs]
    if x0 is None:
        x0 = [0.1] * len(vars_syms)
    sol_vec = sp.nsolve(eqs_num, list(vars_syms), x0, tol=1e-12, maxsteps=100)
    return {v: float(val) for v, val in zip(vars_syms, sol_vec)}

def param_sweep_1d(game: GameModel, stage: Stage, vary_param: str, start: float, stop: float, steps: int,
                   fixed_params: Dict[str, float], x0: Optional[List[float]]=None):
    grid = np.linspace(start, stop, steps)
    rows = []
    last_sol = x0
    for val in grid:
        pv = dict(fixed_params)
        pv[vary_param] = float(val)
        try:
            sol = numeric_solve_given_params(game, stage, pv, x0=last_sol)
            last_sol = list(sol.values())
            row = {"param": val, **{str(k): v for k, v in sol.items()}}
            rows.append(row)
        except Exception:
            row = {"param": val}
            for pname in stage.players:
                for vn in game.players[pname].var_names:
                    row[vn] = None
            rows.append(row)
    return pd.DataFrame(rows)
