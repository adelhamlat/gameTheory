
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import sympy as sp

@dataclass
class Parameter:
    name: str
    value: Optional[float] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None

@dataclass
class Player:
    name: str
    var_names: List[str]
    utility_text: str
    utility_expr: Optional[sp.Expr] = None

@dataclass
class Constraint:
    kind: str
    expr_text: Optional[str] = None
    var: Optional[str] = None
    bounds: Optional[Tuple[Optional[float], Optional[float]]] = None

@dataclass
class Stage:
    players: List[str]

@dataclass
class GameModel:
    players: Dict[str, Player]
    variables: Dict[str, sp.Symbol]
    parameters: Dict[str, Parameter]
    param_symbols: Dict[str, sp.Symbol]
    constraints: List[Constraint]
    stages: List[Stage]

    def all_player_vars(self) -> List[sp.Symbol]:
        out = []
        for p in self.players.values():
            out += [self.variables[v] for v in p.var_names]
        return out
