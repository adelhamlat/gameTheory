# gametheory_symbolic/solvers/partial_symbolic.py
from __future__ import annotations
from typing import Dict, List, Tuple, Any
import sympy as sp

class PartialStageResult:
    """
    Résultats 'meilleur effort' pour un étage :
      - solved: solutions fermées var -> expr
      - unsolved_eqs: équations d'optimalité (FOC) restantes sous forme implicite
      - second_derivs: dérivées secondes par joueur (pour vérif. de concavité simple)
      - var_to_player: mapping Symbole de variable -> nom de joueur (pour rendu UI)
    """
    def __init__(
        self,
        stage_name: str,
        solved: Dict[sp.Symbol, sp.Expr],
        unsolved_eqs: List[sp.Eq],
        second_derivs: Dict[str, Dict[str, sp.Expr]],
        var_to_player: Dict[sp.Symbol, str]
    ):
        self.stage_name = stage_name
        self.solved = solved
        self.unsolved_eqs = unsolved_eqs
        self.second_derivs = second_derivs
        self.var_to_player = var_to_player

# ---------- Calcul différentiel (FOC & d²) ----------

def _derivatives_for_player(U: sp.Expr, vars_syms: List[sp.Symbol]) -> Tuple[List[sp.Eq], Dict[str, sp.Expr]]:
    """Construit les équations de premier ordre (FOC) et les dérivées secondes pour un joueur."""
    foc_eqs: List[sp.Eq] = []
    d2: Dict[str, sp.Expr] = {}
    for v in vars_syms:
        dU = sp.diff(U, v)
        foc_eqs.append(sp.Eq(sp.simplify(dU), 0))
        d2[str(v)] = sp.simplify(sp.diff(U, v, v))
    return foc_eqs, d2

# ---------- Helpers d'algèbre ----------

def _reduce_equations(eqs: List[sp.Eq], subsd: Dict[sp.Symbol, sp.Expr]) -> List[sp.Eq]:
    """
    Réduit les équations après substitution. Ne conserve que celles qui ne deviennent pas identiquement nulles.
    """
    if not eqs:
        return []
    if not subsd:
        return list(eqs)
    still: List[sp.Eq] = []
    for eq in eqs:
        lhs = sp.simplify(eq.lhs.subs(subsd))
        if not sp.simplify(lhs) == 0:
            still.append(sp.Eq(lhs, 0))
    return still

def _choose_nonnegative_root(candidates: List[sp.Expr]) -> sp.Expr | None:
    """
    Prefer a root that is (a) real and (b) >= 0 under assumptions.
    If SymPy can't decide symbolically, use numeric sampling with positive values
    for all free symbols to choose the nonnegative branch.
    """
    if not candidates:
        return None

    # 1) First pass: purely symbolic filters
    symbolic_ok: List[sp.Expr] = []
    for c in candidates:
        if c.is_real is False:
            continue
        try:
            ge0 = sp.simplify(sp.Ge(c, 0))
            if ge0 is True:
                symbolic_ok.append(sp.simplify(c))
                continue
        except Exception:
            pass
        if getattr(c, "is_nonnegative", None) is True:
            symbolic_ok.append(sp.simplify(c))

    if symbolic_ok:
        # pick the simplest symbolically nonnegative root
        return min(symbolic_ok, key=lambda e: sp.count_ops(e))

    # 2) Numeric sampling fallback on the remaining candidates
    def score_candidate(expr: sp.Expr) -> tuple[int, float]:
        """
        Return (nonneg_count, avg_value) using several positive samples.
        Higher nonneg_count is better; if tie, larger avg_value is preferred.
        """
        # choose a few deterministic positive samples for robustness
        samples = [1.0, 1.5, 2.0, 3.0]
        syms = list(expr.free_symbols)

        nonneg_hits = 0
        vals = []
        for s in samples:
            subs = {sym: float(s) for sym in syms}  # all symbols positive
            try:
                val = sp.N(expr.subs(subs))
                # discard complex
                if val.is_real is False:
                    continue
                v = float(val)
                vals.append(v)
                if v >= -1e-12:  # tolerant >= 0
                    nonneg_hits += 1
            except Exception:
                # fail closed for this sample
                continue

        avg = sum(vals) / len(vals) if vals else float("-inf")
        return nonneg_hits, avg

    scored = []
    for c in candidates:
        if c.is_real is False:
            continue
        scored.append((score_candidate(c), sp.simplify(c)))

    if not scored:
        return None

    # Sort by best (nonneg_count desc, avg_value desc), then by simplicity
    scored.sort(key=lambda item: (item[0][0], item[0][1], -1 * sp.count_ops(item[1])), reverse=True)
    best = scored[0][1]
    return best

# ---------- Résolution 'meilleur effort' ----------

def try_solve_system(eq_list: List[sp.Eq],
                     var_list: List[sp.Symbol]) -> Tuple[Dict[sp.Symbol, sp.Expr], List[sp.Eq]]:
    """
    (1) Try global solve → may return multiple dicts (multiple roots).
        For each variable, pick a real & >=0 root if possible (symbolically), otherwise
        use numeric sampling to choose the nonnegative branch.
    (2) For variables still unresolved, try per-variable solveset and filter roots similarly.
    """
    solved: Dict[sp.Symbol, sp.Expr] = {}
    remaining: List[sp.Eq] = list(eq_list or [])

    # --- 1) Global solve with multiple candidates
    global_candidates: List[Dict[sp.Symbol, sp.Expr]] = []
    try:
        if remaining and var_list:
            sols = sp.solve(remaining, var_list, dict=True, simplify=True)
            if sols:
                global_candidates = sols if isinstance(sols, list) else [sols]
    except Exception:
        pass

    if global_candidates:
        # For each variable, collect candidates across dicts, then choose a nonneg root
        for v in var_list:
            cands_v: List[sp.Expr] = []
            for d in global_candidates:
                if v in d:
                    try:
                        cands_v.append(sp.simplify(d[v]))
                    except Exception:
                        cands_v.append(d[v])
            if cands_v:
                pick = _choose_nonnegative_root(cands_v)
                if pick is None:
                    # No nonneg branch found → keep the first (don’t drop info)
                    pick = cands_v[0]
                solved[v] = sp.simplify(pick)
        if solved:
            remaining = _reduce_equations(remaining, solved)

    # --- 2) Per-variable fallback
    for v in var_list:
        if v in solved:
            continue
        eqs_v = [eq for eq in remaining if eq.lhs.has(v)]
        if not eqs_v:
            continue
        try:
            S = sp.solveset(sp.Eq(eqs_v[0].lhs, 0), v, domain=sp.S.Complexes)
            # Normalise candidates
            cands: List[sp.Expr] = []
            if isinstance(S, sp.FiniteSet):
                cands = [sp.simplify(r) for r in list(S)]
            elif S is sp.S.EmptySet:
                cands = []

            pick = _choose_nonnegative_root(cands) if cands else None
            if pick is None and cands:
                pick = cands[0]
            if pick is not None:
                solved[v] = sp.simplify(pick)
                remaining = _reduce_equations(remaining, {v: solved[v]})
        except Exception:
            continue

    if solved:
        remaining = _reduce_equations(remaining, solved)

    return solved, remaining

def solve_stage_partial(stage_name: str,
                        players_in_stage: List[Dict[str, Any]],
                        sym_locals: Dict[str, sp.Symbol]) -> PartialStageResult:
    """
    Construit les FOC de l'étage et résout en mode 'meilleur effort'.
    Retourne les solutions fermées trouvées, les FOC implicites restantes,
    les dérivées secondes par joueur, et le mapping var->joueur.
    """
    all_eqs: List[sp.Eq] = []
    all_vars: List[sp.Symbol] = []
    second_derivs: Dict[str, Dict[str, sp.Expr]] = {}
    var_to_player: Dict[sp.Symbol, str] = {}

    for p in players_in_stage:
        pname = p.get("name", "?")
        util_txt = (p.get("utility") or "").strip()
        if not util_txt:
            second_derivs[pname] = {}
            continue
        U = sp.sympify(util_txt, locals=sym_locals)
        p_vars = [sym_locals[vn] for vn in p.get("vars", []) if vn in sym_locals]
        foc_eqs, d2 = _derivatives_for_player(U, p_vars)
        all_eqs.extend(foc_eqs)
        all_vars.extend(p_vars)
        for v in p_vars:
            var_to_player[v] = pname
        second_derivs[pname] = d2

    solved, remaining = try_solve_system(all_eqs, all_vars)
    return PartialStageResult(stage_name, solved, remaining, second_derivs, var_to_player)

def sequential_backward_partial(stages: List[Dict[str, Any]],
                                sym_locals: Dict[str, sp.Symbol]) -> List[PartialStageResult]:
    """
    Backward induction 'meilleur effort'.
    À chaque étage (en partant du dernier), on résout partiellement, on accumule
    les substitutions trouvées, puis on remonte aux étages précédents.
    """
    results: List[PartialStageResult] = []
    subs_acc: Dict[sp.Symbol, sp.Expr] = {}

    for stage in reversed(stages):
        # locals avec substitutions connues
        loc = dict(sym_locals)
        if subs_acc:
            for k, v in list(loc.items()):
                if isinstance(v, sp.Symbol) and v in subs_acc:
                    loc[k] = subs_acc[v]

        r = solve_stage_partial(stage.get("name", "Stage"), stage.get("players", []), loc)
        results.append(r)

        # alimenter les substitutions pour l'étage précédent
        for v, expr in r.solved.items():
            subs_acc[v] = expr

    # Remettre dans l'ordre naturel (leaders -> followers)
    return list(reversed(results))
