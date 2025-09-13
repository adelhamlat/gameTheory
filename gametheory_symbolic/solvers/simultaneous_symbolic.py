
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import sympy as sp
from ..core.types import GameModel, Stage
from ..algebra.calculus import gradient, second_derivatives
from ..algebra.solve_sym import try_solve_symbolic

class SolveError(Exception): ...

def build_foc_equations_symbolic(game: GameModel, stage: Stage) -> Tuple[List[sp.Eq], List[sp.Symbol]]:
    stage_vars: List[sp.Symbol] = []
    for pname in stage.players:
        player = game.players[pname]
        stage_vars += [game.variables[v] for v in player.var_names]
    eqs: List[sp.Eq] = []
    for pname in stage.players:
        player = game.players[pname]
        own_vars = [game.variables[v] for v in player.var_names]
        grad = gradient(player.utility_expr, own_vars)
        for g in grad:
            eqs.append(sp.Eq(sp.simplify(g), 0))
    return eqs, stage_vars

def check_soc_second_derivatives(game: GameModel, stage: Stage, sol_map: Dict[sp.Symbol, sp.Expr]):
    out = {}
    for pname in stage.players:
        player = game.players[pname]
        own_vars = [game.variables[v] for v in player.var_names]
        d2 = second_derivatives(player.utility_expr, own_vars)
        d2_sub = {v: sp.simplify(expr.subs(sol_map)) for v, expr in d2.items()}
        out[pname] = d2_sub
    return out

def solve_stage_symbolic(game: GameModel, stage: Stage):
    eqs, vars_syms = build_foc_equations_symbolic(game, stage)
    sym_sol = try_solve_symbolic(eqs, vars_syms)
    if sym_sol is None:
        raise SolveError("No closed-form solution found for this stage (symbolic).")
    soc = check_soc_second_derivatives(game, stage, sym_sol)
    return sym_sol, soc
