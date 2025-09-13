
from __future__ import annotations
from typing import Dict, List
import sympy as sp

class UnsupportedFunctionError(Exception): ...
class UnknownSymbolError(Exception): ...

_ALLOWED_FUNCS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "Abs": sp.Abs,
    "Min": sp.Min, "Max": sp.Max
}

def safe_functions() -> Dict[str, object]:
    return dict(_ALLOWED_FUNCS)

def build_symbols(var_names: List[str], param_names: List[str]):
    variables = {n: sp.Symbol(n, real=True) for n in var_names}
    params = {n: sp.Symbol(n, real=True) for n in param_names}
    return variables, params

def make_expr(text: str, locals_map: Dict[str, object]) -> sp.Expr:
    expr = sp.sympify(text, locals=locals_map, evaluate=True)
    unknown = [s for s in expr.free_symbols if s.name not in locals_map]
    if unknown:
        raise UnknownSymbolError(f"Unknown symbols in expression: {[s.name for s in unknown]}")
    if "__" in text:
        raise UnsupportedFunctionError("Dunder access not allowed")
    return expr
