# ui/app2.py
from __future__ import annotations

import os, sys
from typing import Dict, List, Set, Tuple, Optional, Any
import re
import html

# --- sys.path for local package imports (if running from repo root) ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

# Core + solvers
from gametheory_symbolic.core.model import build_model, ModelValidationError
from gametheory_symbolic.solvers.simulate_numeric import param_sweep_1d
from gametheory_symbolic.solvers.partial_symbolic import (
    PartialStageResult,
    sequential_backward_partial,
)
from gametheory_symbolic.mathpix_ocr import ocr_image_file, extract_single_equation
# =========================
# --------- THEME / CSS ----
# =========================
st.set_page_config(page_title="Game Theory Model Solver", page_icon="🎯", layout="wide")

st.markdown("""
<style>
/* --- Result card + wrap long expressions --- */
div[data-testid="stMarkdownContainer"] .result {
  border: 1px solid #e5e7eb;
  background: #0b1020;
  color: #e5e7eb;
  border-radius: 12px;
  padding: .8rem 1rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  max-width: 100%;
  overflow-x: hidden;
}
div[data-testid="stMarkdownContainer"] .result pre,
div[data-testid="stMarkdownContainer"] .result code {
  white-space: pre-wrap !important;
  word-break: break-word !important;
  overflow-wrap: anywhere !important;
  line-break: anywhere !important;
  margin: 0;
  max-width: 100% !important;
}

:root {
  --bg:#ffffff; --fg:#0f172a; --muted:#64748b; --line:#e5e7eb;
  --card:#ffffff; --cardBorder:#e5e7eb;
  --primary:#1d4ed8; --success:#16a34a; --soft:#f8fafc;
}
html, body { margin:0; padding:0; }
.stApp { background:var(--bg); color:var(--fg); }
.block-container { max-width:1220px; padding-top:.8rem; }
h1.app-title { text-align:center; margin:.2rem 0 1.1rem 0; line-height:1.1; }

.block-container > div:empty,
.block-container section:empty,
.stColumn > div:empty,
div[aria-live="polite"]:empty { display:none !important; height:0 !important; }

/* Cards */
.card { border:1px solid var(--cardBorder); border-radius:14px; padding:1rem 1.2rem; background:var(--card); box-shadow:0 1px 2px rgba(0,0,0,.04); }

/* Player card */
.player-card {
  background:var(--soft);
  border:1px solid var(--cardBorder);
  border-radius:12px;
  padding:1rem;
  margin-bottom:.9rem;
}
.player-title { font-weight:800; font-size:1.05rem; margin-bottom:.35rem; }

/* Row: Name | Variables | Action (trash) */
.player-head { display:flex; gap:.6rem; align-items:flex-end; }
.head-name { flex:1 1 0; }
.head-vars { flex:1 1 0; }
.head-actions { width:56px; }

.icon-btn button { padding:.45rem .5rem; border-radius:10px; width:100%; }

.mono textarea, .mono input { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.section-title { font-weight:800; font-size:1.05rem; margin:.2rem 0 .6rem 0; }

.badges { display:flex; gap:.35rem; flex-wrap:wrap; }
.badge { background:#f1f5f9; color:#0f172a; border:1px solid var(--cardBorder);
         border-radius:999px; padding:.15rem .6rem; font-size:.85rem; }

.primary-btn button, .success-btn button {
  width:100%; font-weight:700; padding:.7rem .9rem; border-radius:12px;
}
.primary-btn button { background:var(--primary); color:#fff; border-color:var(--primary); }
.success-btn button { background:var(--success); color:#fff; border-color:var(--success); }

.result { border:1px solid var(--cardBorder); background:#0b1020; color:#e5e7eb;
          border-radius:12px; padding:.8rem 1rem; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.result h4, .result h5 { margin:.2rem 0 .4rem 0; color:#cbd5e1; }

/* Sidebar */
.sidebar-card { border-top:1px solid var(--line); padding-top:.8rem; margin-top:.8rem; }

/* Placeholder */
.placeholder {
  border:1px dashed var(--line);
  border-radius:12px;
  padding:1rem;
  color:#94a3b8;
  text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='app-title'>Game-Theory Model Solver </h1>", unsafe_allow_html=True)

# =========================
# ------ STATE INIT -------
# =========================
def init_state():
    if "players" not in st.session_state:
        # Dict style for easy renaming & ordering
        st.session_state.players: Dict[str, Dict[str, Any]] = {
            "Gov": {"vars_line":"T X", "vars":["T","X"],
                    "utility":"-(g_T*(1+alpha*e_t)*t)/((1+alpha*e_t)*t+T) - (g_x*(1+beta*e_x)*x)/((1+beta*e_x)*x+X)"},
            "TG":  {"vars_line":"t x", "vars":["t","x"],
                    "utility":"(mu_t*(1+alpha*e_t)*t)/((1+alpha*e_t)*t+T) + (mu_x*(1+beta*e_x)*x)/((1+beta*e_x)*x+X)"},
            "P2":  {"vars_line":"e_t", "vars":["e_t"],
                    "utility":"-k_e_t*e_t + ((1+alpha*e_t)*t)/((1+alpha*e_t)*t+T)*P_TG"},
            "P3":  {"vars_line":"e_x", "vars":["e_x"],
                    "utility":"-k_e_x*e_x + (((1+beta*e_x)*x)/(((1+beta*e_x)*x)+X))*P_x"},
        }
    if "player_order" not in st.session_state:
        st.session_state.player_order: List[str] = list(st.session_state.players.keys())
    if "sequential" not in st.session_state:
        st.session_state.sequential = True  # default: sequential to match your use-case
    if "stages" not in st.session_state:
        # Default stages: P2 & P3 → TG → Gov
        st.session_state.stages: List[List[str]] = [["P2","P3"], ["TG"], ["Gov"]]
    if "constraints" not in st.session_state:
        st.session_state.constraints: List[str] = []
    if "sim_open" not in st.session_state:
        st.session_state.sim_open = False
    if "sim_cfg" not in st.session_state:
        st.session_state.sim_cfg = {"vary": None, "start": 10.0, "stop": 100.0, "steps": 10, "fixed": {}}
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""

init_state()

def sync_player_order():
    names = list(st.session_state.players.keys())
    order = [n for n in st.session_state.player_order if n in names]
    for n in names:
        if n not in order:
            order.append(n)
    st.session_state.player_order = order

sync_player_order()

# =========================
# -- PARAM DETECTION SAFE --
# =========================
RESERVED_NAMES = {
    # SymPy reserved/common
    "beta", "gamma", "zeta", "eta", "theta", "lambda", "Lambda",
    "mu", "nu", "pi", "Pi", "E", "I", "O", "DiracDelta"
}
IDENT_RE = re.compile(r"[A-Za-z_]\w*")

def normalize_vars_line(s: str) -> List[str]:
    return [x for x in s.replace(",", " ").split() if x]

def infer_parameters_from_utilities(players: Dict[str, Dict]) -> List[str]:
    # 1) all declared variables
    var_set: Set[str] = set()
    for pdata in players.values():
        var_set |= set(pdata.get("vars", []))

    # 2) lexical scan of all identifiers in utilities
    all_names: Set[str] = set()
    for pdata in players.values():
        txt = (pdata.get("utility", "") or "")
        for name in IDENT_RE.findall(txt):
            all_names.add(name)

    # 3) candidates = seen - variables
    candidates = sorted(n for n in all_names if n not in var_set)

    # 4) build safe locals forcing every name to a Symbol
    safe_locals = {v: sp.Symbol(v, real=True) for v in var_set}
    for n in candidates:
        if (n in RESERVED_NAMES) or (n not in safe_locals):
            safe_locals[n] = sp.Symbol(n, real=True)

    # 5) confirm with sympify
    params: Set[str] = set()
    for pdata in players.values():
        txt = (pdata.get("utility", "") or "").strip()
        if not txt:
            continue
        try:
            expr = sp.sympify(txt, locals=safe_locals)
            for s in expr.free_symbols:
                if s.name not in var_set:
                    params.add(s.name)
        except Exception:
            # fallback lexical
            for n in IDENT_RE.findall(txt):
                if n not in var_set:
                    params.add(n)
    return sorted(params)

def collect_all_variables(players: Dict[str, Dict]) -> List[str]:
    out: Set[str] = set()
    for pdata in players.values():
        out |= set(pdata.get("vars", []))
    return sorted(out)

def current_param_names() -> List[str]:
    return infer_parameters_from_utilities(st.session_state.players)

# =========================
# ---- SYMBOL LOCALS -------
# =========================
def build_sym_locals_for_partial(players: Dict[str, Dict], param_names: List[str]) -> Dict[str, sp.Symbol]:
    """
    Create SymPy symbols with assumptions for the PARTIAL solver:
      - decision variables: nonnegative (>=0)
      - parameters: positive (>0)
    """
    var_names = collect_all_variables(players)
    sym_locals: Dict[str, sp.Symbol] = {}
    for n in var_names:
        sym_locals[n] = sp.Symbol(n, real=True, nonnegative=True)
    for n in param_names:
        sym_locals[n] = sp.Symbol(n, real=True, positive=True)
    return sym_locals

# =========================
# ------- WRAP HELPERS ----
# =========================
ZERO_WIDTH_SPACE = "\u200b"
BREAK_AFTER = re.compile(r'([+\-*/^=,:])')   # after operators & commas
BREAK_PAREN = re.compile(r'([()])')          # around parentheses

def make_wrappable(s: str) -> str:
    s = BREAK_AFTER.sub(lambda m: m.group(1) + ZERO_WIDTH_SPACE, s)
    s = BREAK_PAREN.sub(lambda m: ZERO_WIDTH_SPACE + m.group(1) + ZERO_WIDTH_SPACE, s)
    return s

def render_result_block(title: str, lines: List[str]):
    wrapped = [make_wrappable(line) for line in lines]
    text = html.escape("\n".join(wrapped))
    st.markdown(
        f"<div class='result'><h4>{html.escape(title)}</h4><pre>{text}</pre></div>",
        unsafe_allow_html=True
    )

# =========================
# --- SPEC / STAGES BUILD --
# =========================
def build_spec() -> Dict:
    players_list = []
    for name in st.session_state.player_order:
        pdata = st.session_state.players[name]
        if not name.strip():
            continue
        utility = (pdata.get("utility", "") or "").strip()
        if not utility:
            continue
        players_list.append({"name": name, "vars": pdata.get("vars", []), "utility": utility})

    inferred_params = [{"name": p} for p in infer_parameters_from_utilities(st.session_state.players)]
    constraints = [{"kind": "raw", "expr": c.strip()} for c in st.session_state.constraints if c.strip()]

    if st.session_state.sequential:
        stages = [{"players": s} for s in st.session_state.stages if s]
    else:
        stages = [{"players": [p["name"] for p in players_list]}] if players_list else []

    return {"players": players_list, "parameters": inferred_params, "constraints": constraints, "stages": stages}

def partial_stages_from_spec(spec: Dict) -> List[Dict[str, Any]]:
    """
    Convert 'spec' into stages for sequential_backward_partial:
      [{ 'name': 'Stage i', 'players': [ {'name', 'vars', 'utility'} ] }]
    """
    # index players by name
    pmap = {p["name"]: p for p in spec.get("players", [])}
    out: List[Dict[str, Any]] = []
    for i, stg in enumerate(spec.get("stages", []), start=1):
        players_objs = []
        for pname in stg.get("players", []):
            if pname in pmap:
                players_objs.append({
                    "name": pmap[pname]["name"],
                    "vars": pmap[pname].get("vars", []),
                    "utility": pmap[pname].get("utility", "")
                })
        out.append({"name": f"Stage {i}", "players": players_objs})
    return out

# ---------- Parsing constraints (for simulation filtering) ----------
OPS = [">=", "<=", ">", "<", "="]

def parse_constraint(raw: str, symbols_map: Dict[str, sp.Symbol]) -> Optional[Tuple[sp.Expr, str, sp.Expr]]:
    s = (raw or "").strip()
    if not s:
        return None
    op_found = None
    for op in OPS:
        if op in s:
            op_found = op
            break
    if not op_found:
        try:
            lhs = sp.sympify(s, locals=symbols_map)
            rhs = sp.Integer(0)
            return lhs, "=", rhs
        except Exception:
            return None
    parts = s.split(op_found)
    if len(parts) != 2:
        return None
    left, right = parts[0].strip(), parts[1].strip()
    try:
        lhs = sp.sympify(left, locals=symbols_map)
        rhs = sp.sympify(right, locals=symbols_map)
        return lhs, op_found, rhs
    except Exception:
        return None

# =========================
# ------- SIDEBAR ---------
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if st.button("Reset model"):
        st.session_state.players = {
            "J1": {"vars_line":"q1", "vars":["q1"], "utility":""}
        }
        st.session_state.player_order = ["J1"]
        st.session_state.sequential = False
        st.session_state.stages = []
        st.session_state.constraints = []
        st.session_state.sim_open = False
        st.session_state.sim_cfg = {"vary": None, "start": 10.0, "stop": 100.0, "steps": 10, "fixed": {}}
        st.rerun()

    st.markdown("<div class='sidebar-card'></div>", unsafe_allow_html=True)

    st.markdown("#### Play order")
    st.session_state.sequential = st.radio(
        "Mode", options=["Simultaneous", "Sequential"],
        index=1 if st.session_state.sequential else 0,
        horizontal=True, label_visibility="collapsed"
    ) == "Sequential"

    if st.session_state.sequential:
        if not st.session_state.stages:
            # initialize with first player only
            init_players = list(st.session_state.player_order)[:1]
            st.session_state.stages = [init_players]
        for si, stage in enumerate(list(st.session_state.stages)):
            st.caption(f"Stage {si+1}")
            options = list(st.session_state.player_order)
            selected = st.multiselect("Players", options, default=stage,
                                      key=f"stage_{si}", label_visibility="collapsed")
            st.session_state.stages[si] = selected
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("Add stage", key=f"add_stage_{si}"):
                    st.session_state.stages.insert(si+1, [])
                    st.rerun()
            with sc2:
                if st.button("Remove stage", key=f"del_stage_{si}"):
                    st.session_state.stages.pop(si)
                    st.rerun()

    st.markdown("<div class='sidebar-card'></div>", unsafe_allow_html=True)

    st.markdown("#### Constraints")
    st.caption("Write equations/inequalities (e.g., `q1 >= 0`, `R = a*q1 + b*q2`).")
    for i, c in enumerate(list(st.session_state.constraints)):
        cc1, cc2 = st.columns([4, 1])
        with cc1:
            newc = st.text_input("Constraint", value=c, key=f"constraint_{i}",
                                 label_visibility="collapsed",
                                 placeholder="e.g., q1 >= 0  or  R = a*q1 + b*q2")
            if newc != c:
                st.session_state.constraints[i] = newc
        with cc2:
            if st.button("🗑️", key=f"del_constraint_{i}", help="Delete"):
                st.session_state.constraints.pop(i)
                st.rerun()
    if st.button("Add constraint"):
        st.session_state.constraints.append("")
        st.rerun()

# =========================
# ---------- UI -----------
# =========================
left, right = st.columns([1.1, 0.9])

# ============================================================
# Left panel — Players + actions + detected parameters
# ============================================================
def unique_name(base: str) -> str:
    if base not in st.session_state.players:
        return base
    i = 2
    while f"{base}{i}" in st.session_state.players:
        i += 1
    return f"{base}{i}"

def rename_player(old: str, new: str):
    if not new or new == old:
        return
    if new in st.session_state.players:
        return
    st.session_state.players[new] = st.session_state.players.pop(old)
    st.session_state.stages = [[(new if x == old else x) for x in stage] for stage in st.session_state.stages]
    st.session_state.player_order = [new if x == old else x for x in st.session_state.player_order]

with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Players</div>", unsafe_allow_html=True)

    # Players list (titles: Player 1/2/... + editable Name)
    for idx, pname in enumerate(list(st.session_state.player_order)):
        pdata = st.session_state.players[pname]
        st.markdown("<div class='player-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='player-title'>Player {idx+1}</div>", unsafe_allow_html=True)

        c_name, c_vars, c_actions = st.columns([3, 3, 1])
        with c_name:
            new_name = st.text_input("Name", value=pname, key=f"name_{pname}",
                                     placeholder="e.g., A", label_visibility="visible")
        with c_vars:
            vars_line = st.text_input("Variables", value=pdata.get("vars_line", ""),
                                      key=f"vars_{pname}", placeholder="e.g., q1 q2",
                                      label_visibility="visible")
        with c_actions:
            if st.button("🗑️", key=f"del_{pname}", help="Delete player", use_container_width=True):
                st.session_state.players.pop(pname, None)
                st.session_state.player_order = [n for n in st.session_state.player_order if n != pname]
                st.session_state.stages = [[x for x in stage if x != pname] for stage in st.session_state.stages]
                st.rerun()

        # Apply name & variables
        if new_name != pname:
            if new_name and new_name not in st.session_state.players:
                rename_player(pname, new_name)
                pname = new_name
                pdata = st.session_state.players[pname]
                st.rerun()

        pdata["vars_line"] = vars_line
        pdata["vars"] = normalize_vars_line(vars_line)

        # ---- En-tête: "Utility function" + bouton OCR à droite ----
        u_hdr, u_btn = st.columns([6, 1])
        with u_hdr:
            st.markdown("**Utility function**")
        with u_btn:
            if st.button("📷 OCR", key=f"ocr_btn_{pname}", help="Import from image"):
                st.session_state[f"ocr_open_{pname}"] = True
        
        # --- Clés par joueur
        util_key  = f"util_{pname}"
        force_key = f"util_force_load_{pname}"  # recharger depuis le modèle après OCR (une seule fois)
        
        # 1) Initialiser la valeur du widget à partir du modèle la toute première fois
        if util_key not in st.session_state:
            st.session_state[util_key] = pdata.get("utility", "")
        
        # 2) Recharger depuis le modèle UNIQUEMENT si un OCR vient d'arriver
        elif st.session_state.get(force_key, False):
            st.session_state[util_key] = pdata.get("utility", "")
            st.session_state[force_key] = False  # on a consommé le reload
        
        # Zone d’édition liée à util_key (pas de 'value=')
        pdata["utility"] = st.text_area(
            "Utility function",
            key=util_key,
            height=100,
            placeholder="e.g., (a - b*(q1+q2) - c1)*q1",
            label_visibility="visible"
        )
        
        # Propager l’édition utilisateur vers le modèle (et seulement si ça change)
        new_val = st.session_state[util_key]
        if new_val != st.session_state.players[pname].get("utility", ""):
            st.session_state.players[pname]["utility"] = new_val
        
        # Si l'utilisateur a cliqué sur OCR : uploader minimal
        if st.session_state.get(f"ocr_open_{pname}", False):
            import tempfile, os
            from mathpix_ocr import ocr_image_file, extract_single_equation  # assure-toi que l'import est dispo
        
            uploader = st.file_uploader(
                "Choose an image (PNG/JPG/BMP/TIFF/PDF)",
                type=["png", "jpg", "jpeg", "bmp", "tiff", "pdf"],
                key=f"ocr_upl_{pname}"
            )
        
            cols_ocr = st.columns([1, 1, 3])
            with cols_ocr[0]:
                replace_current = st.checkbox("Replace", value=True, key=f"ocr_replace_{pname}")
            with cols_ocr[1]:
                if st.button("Close", key=f"ocr_close_{pname}"):
                    st.session_state[f"ocr_open_{pname}"] = False
                    st.rerun()
        
            if uploader is not None:
                suffix = os.path.splitext(uploader.name)[1] or ".png"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploader.read())
                    tmp.flush()
                    tmp_path = tmp.name
                try:
                    resp = ocr_image_file(tmp_path)
                    eq = extract_single_equation(resp)
                    if not eq:
                        st.warning("No valid equation detected in the OCR result.")
                    else:
                        if replace_current:
                            new_text = eq
                        else:
                            prev = st.session_state.players[pname].get("utility", "")
                            new_text = (prev + ("\n" if prev else "") + eq).strip()
        
                        # Mettre à jour le modèle, puis demander au widget de se recharger UNE fois
                        st.session_state.players[pname]["utility"] = new_text
                        st.session_state[force_key] = True
        
                        st.success("Equation inserted from OCR.")
                        st.session_state[f"ocr_open_{pname}"] = False
                        st.rerun()
                finally:
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

    # Add player (bottom)
    if st.button("➕ Add player", key="add_player_btn_bottom", use_container_width=True):
        base = "J"
        newn = unique_name(base + "2")
        st.session_state.players[newn] = {"vars_line": "", "vars": [], "utility": ""}
        st.session_state.player_order.append(newn)
        st.rerun()

    # Primary actions
    act_l, act_r = st.columns(2)
    with act_l:
        solve_clicked = st.button("⚙️ Solve (symbolic - partial)", key="solve_btn", use_container_width=True)
    with act_r:
        sim_clicked = st.button("📈 Simulation (numeric)", key="sim_btn", use_container_width=True)
    if sim_clicked:
        st.session_state.sim_open = True  # keep simulation pane open across reruns

    # Detected parameters (from utilities)
    st.markdown("<div class='section-title'>Detected parameters</div>", unsafe_allow_html=True)
    inferred = current_param_names()
    if inferred:
        st.markdown("<div class='badges'>" + "".join([f"<span class='badge'>{p}</span>" for p in inferred]) + "</div>", unsafe_allow_html=True)
    else:
        st.write("—")

    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# Right panel — Results (solve + simulation)
# ============================================================
# Wrapping helpers for result blocks
def _owner_from_session(var_symbol: sp.Symbol) -> str:
    vname = str(var_symbol)
    for p in st.session_state.get("players", {}).items():
        pass
    for name, pdata in st.session_state.get("players", {}).items():
        if vname in (pdata.get("vars") or []):
            return name
    return "Player"

with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    showed_anything = False

    # ---------------- Solve (symbolic - partial) ----------------
    if 'solve_clicked' in locals() and solve_clicked:
        showed_anything = True
        try:
            # Build 'spec' from UI then turn into partial stages
            spec = build_spec()
            # Build locals with assumptions for partial solver
            param_names = [p["name"] for p in spec["parameters"]]
            sym_locals_partial = build_sym_locals_for_partial(st.session_state.players, param_names)
            stages_partial = partial_stages_from_spec(spec)

            st.subheader("Solve — Symbolic (Partial, with backward induction)")

            results: List[PartialStageResult] = sequential_backward_partial(stages_partial, sym_locals_partial)

            # Render per stage
            for r in results:
                st.markdown(f"#### {r.stage_name}")

                # Group closed-form solutions by player
                if r.solved:
                    by_player: Dict[str, List[str]] = {}
                    for v, expr in r.solved.items():
                        pname = r.var_to_player.get(v, _owner_from_session(v)) if hasattr(r, "var_to_player") else _owner_from_session(v)
                        line = f"{str(v)} = {sp.simplify(expr)}"
                        by_player.setdefault(pname, []).append(line)
                    # stable order: as in spec
                    spec_order = [p["name"] for p in spec["players"]]
                    for pname in spec_order:
                        lines = by_player.get(pname, [])
                        if lines:
                            render_result_block(f"Closed-form solutions — {pname}", lines)
                    # any other (unlikely)
                    for pname, lines in by_player.items():
                        if pname not in spec_order:
                            render_result_block(f"Closed-form solutions — {pname}", lines)
                else:
                    st.info("No closed-form solution found for this stage.")

                # FOC still unsolved (implicit)
                if r.unsolved_eqs:
                    foc_lines = [str(sp.simplify(eq.lhs)) + " = 0" for eq in r.unsolved_eqs]
                    render_result_block("Implicit first-order conditions (unsolved)", foc_lines)

                # Second derivatives (concavity check)
                d2_lines: List[str] = []
                for pname, d2map in r.second_derivs.items():
                    for vname, d2 in d2map.items():
                        d2_lines.append(f"{pname}: d²U/d{vname}² = {sp.simplify(d2)}")
                if d2_lines:
                    with st.expander("Second derivatives (concavity check)"):
                        render_result_block("Second derivatives", d2_lines)

        except ModelValidationError as e:
            st.session_state.last_error = str(e); st.error(st.session_state.last_error)
        except Exception as e:
            st.session_state.last_error = str(e); st.exception(e)

    # ---------------- Simulation (numeric) ----------------
    if st.session_state.sim_open:
        showed_anything = True
        try:
            spec = build_spec()
            gm = build_model(spec)  # uses your core model builder
            param_names = [p["name"] for p in spec["parameters"]]

            col_header_l, col_header_r = st.columns([3, 1])
            with col_header_l:
                st.subheader("Simulation (numeric)")
            with col_header_r:
                if st.button("Close ✖️", key="close_sim"):
                    st.session_state.sim_open = False

            if not param_names:
                st.info("No parameters detected in utility functions.")
            else:
                cfg = st.session_state.sim_cfg
                if cfg["vary"] not in param_names:
                    cfg["vary"] = param_names[0]

                with st.form("sim_form", clear_on_submit=False):
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        cfg["vary"] = st.selectbox("Parameter to vary", param_names, index=param_names.index(cfg["vary"]))
                    with c2:
                        cfg["start"] = st.number_input("Start", value=float(cfg["start"]))
                    with c3:
                        cfg["stop"] = st.number_input("End", value=float(cfg["stop"]))
                    cfg["steps"] = st.number_input("Steps", value=int(cfg["steps"]), min_value=2, step=1)

                    st.markdown("**Fixed values**")
                    fixed_vals: Dict[str, float] = {}
                    for pn in param_names:
                        if pn == cfg["vary"]:
                            continue
                        prev = cfg["fixed"].get(pn, "")
                        val_str = st.text_input(f"{pn} =", value=str(prev))
                        if val_str.strip():
                            try:
                                fixed_vals[pn] = float(val_str)
                            except:
                                pass
                        cfg["fixed"][pn] = val_str

                    run_sim = st.form_submit_button("▶️ Run simulation")

                st.session_state.sim_cfg = cfg

                if run_sim:
                    if len(gm.stages) != 1:
                        st.warning("Simulation V1 supports a single simultaneous stage.")
                    else:
                        df = param_sweep_1d(
                            gm, gm.stages[0],
                            cfg["vary"],
                            float(cfg["start"]),
                            float(cfg["stop"]),
                            int(cfg["steps"]),
                            fixed_vals,
                            x0=None
                        )

                        # Apply constraints (filtering)
                        var_names = collect_all_variables(st.session_state.players)
                        all_syms = {n: sp.Symbol(n, real=True) for n in (var_names + param_names)}
                        parsed = []
                        for cstr in st.session_state.constraints:
                            parsed_item = parse_constraint(cstr, all_syms)
                            if parsed_item:
                                parsed.append(parsed_item)

                        def row_satisfies(row: pd.Series) -> bool:
                            subs_num = {}
                            for v in var_names:
                                if v in row.index:
                                    subs_num[all_syms[v]] = float(row[v])
                            subs_num[all_syms[cfg["vary"]]] = float(row["param"])
                            for k, val in fixed_vals.items():
                                subs_num[all_syms[k]] = float(val)
                            for (lhs, op, rhs) in parsed:
                                l = float(sp.N(lhs.subs(subs_num)))
                                r = float(sp.N(rhs.subs(subs_num)))
                                if op == "=" and abs(l - r) > 1e-8:
                                    return False
                                if op == ">=" and not (l >= r - 1e-12):
                                    return False
                                if op == "<=" and not (l <= r + 1e-12):
                                    return False
                                if op == ">" and not (l > r + 1e-12):
                                    return False
                                if op == "<" and not (l < r - 1e-12):
                                    return False
                            return True

                        if parsed:
                            df = df[df.apply(row_satisfies, axis=1)]

                        st.dataframe(df, use_container_width=True)

                        ycols = [c for c in df.columns if c != "param"]
                        if ycols:
                            fig = plt.figure()
                            for c in ycols:
                                plt.plot(df["param"], df[c], label=str(c))
                            plt.xlabel(cfg["vary"]); plt.ylabel("Equilibrium"); plt.legend()
                            st.pyplot(fig)

        except ModelValidationError as e:
            st.error(f"Validation: {e}")
        except Exception as e:
            st.exception(e)

    # Placeholder if nothing shown yet
    if not showed_anything:
        st.markdown("<div class='placeholder'>Results… Run a symbolic solve or a numeric simulation.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
