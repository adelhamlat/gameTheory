
from __future__ import annotations
from typing import Dict, Any
import sympy as sp
from .types import Player, Parameter, Constraint, Stage, GameModel
from ..algebra.parsing import build_symbols, make_expr, safe_functions

class ModelValidationError(Exception):
    pass

def build_model(spec: Dict[str, Any]) -> GameModel:
    players: Dict[str, Player] = {}
    for P in spec.get("players", []):
        players[P["name"]] = Player(name=P["name"],
                                    var_names=P["vars"],
                                    utility_text=P["utility"])
    var_names = sorted({vn for p in players.values() for vn in p.var_names})

    def _coerce_val(v):
        if v is None: return None
        return float(v)
    params_dict = { p["name"]: Parameter(name=p["name"],
                                         value=_coerce_val(p.get("value")),
                                         bounds=tuple(p["bounds"]) if p.get("bounds") else None)
                    for p in spec.get("parameters", []) }

    variables, param_symbols = build_symbols(var_names, list(params_dict.keys()))
    locals_map = {**variables, **param_symbols, **safe_functions()}
    for pl in players.values():
        pl.utility_expr = make_expr(pl.utility_text, locals_map)

    constraints = []
    for C in spec.get("constraints", []):
        if C["kind"] == "eq":
            constraints.append(Constraint(kind="eq", expr_text=C["expr"]))
        elif C["kind"] == "bound":
            constraints.append(Constraint(kind="bound", var=C["var"], bounds=tuple(C["bounds"])))

    stages = [Stage(players=S["players"]) for S in spec.get("stages", [])] or [Stage(players=list(players.keys()))]

    gm = GameModel(players=players, variables=variables, parameters=params_dict,
                   param_symbols=param_symbols, constraints=constraints, stages=stages)
    _validate_model(gm)
    return gm

def _validate_model(gm: GameModel) -> None:
    if not gm.players:
        raise ModelValidationError("No players defined")
    for p in gm.players.values():
        if not p.var_names:
            raise ModelValidationError(f"Player {p.name} has no variables")
    names = set(gm.players.keys())
    for st in gm.stages:
        for nm in st.players:
            if nm not in names:
                raise ModelValidationError(f"Stage references unknown player {nm}")
    all_syms = set(gm.variables.values()) | set(gm.param_symbols.values())
    for p in gm.players.values():
        free = p.utility_expr.free_symbols
        if not free.issubset(all_syms):
            extra = free - all_syms
            raise ModelValidationError(f"Utility of {p.name} references unknown symbols: {extra}")
