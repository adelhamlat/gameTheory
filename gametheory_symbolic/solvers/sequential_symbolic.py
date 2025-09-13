
from __future__ import annotations
from typing import Dict, Any
import sympy as sp
from ..core.types import GameModel
from .simultaneous_symbolic import solve_stage_symbolic

def solve_by_backward_induction_symbolic(game: GameModel) -> Dict[str, Any]:
    players = game.players
    variables = game.variables
    solved_values: Dict[sp.Symbol, Any] = {}
    soc_by_stage = []
    for stage in reversed(game.stages):
        sol, soc = solve_stage_symbolic(game, stage)
        solved_values.update(sol)
        soc_by_stage.append({"stage": stage.players, "soc": soc})
        for p in players.values():
            p.utility_expr = p.utility_expr.subs(sol)
    final_expr = {name: sp.simplify(sym.subs(solved_values)) for name, sym in variables.items()}
    utils = {pname: sp.simplify(p.utility_expr.subs(solved_values)) for pname, p in players.items()}
    return {"variables": final_expr, "utilities": utils, "soc": soc_by_stage}
