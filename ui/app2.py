import os, sys
from typing import Dict, List, Set, Tuple, Optional
import copy

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from gametheory_symbolic.core.model import build_model, ModelValidationError
from gametheory_symbolic.solvers.simultaneous_symbolic import solve_stage_symbolic, SolveError as SymSolveError
from gametheory_symbolic.solvers.sequential_symbolic import solve_by_backward_induction_symbolic
from gametheory_symbolic.solvers.simulate_numeric import param_sweep_1d

# -------------------- Page & Styles --------------------
st.set_page_config(page_title="Game Theory Model Solver ", layout="wide")

st.markdown("""
<style>
:root {
  --bg:#ffffff; --fg:#0f172a; --muted:#64748b; --line:#e5e7eb;
  --card:#ffffff; --cardBorder:#e5e7eb;
  --primary:#1d4ed8; --success:#16a34a; --soft:#f8fafc;
}
html, body { margin:0; padding:0; }
.stApp { background:var(--bg); color:var(--fg); }
.block-container { max-width:1200px; padding-top:.8rem; }
h1.app-title { text-align:center; margin:.2rem 0 1.1rem 0; line-height:1.1; }

/* Hide stray empty containers that can look like rounded separators */
.block-container > div:empty,
.block-container section:empty,
.stColumn > div:empty,
div[aria-live="polite"]:empty {
  display: none !important;
  height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
}

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

/* Compact icon button */
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

st.markdown("<h1 class='app-title'> \n Game Theory Model Solver </h1>", unsafe_allow_html=True)

# -------------------- Session state --------------------
def init_state():
    if "players" not in st.session_state:
        st.session_state.players = {
            "A": {"vars_line":"q1", "vars":["q1"], "utility":"(a - b*(q1+q2) - c1)*q1"},
            "B": {"vars_line":"q2", "vars":["q2"], "utility":"(a - b*(q1+q2) - c2)*q2"},
        }
    if "player_order" not in st.session_state:
        st.session_state.player_order = list(st.session_state.players.keys())
    if "sequential" not in st.session_state:
        st.session_state.sequential = False
    if "stages" not in st.session_state:
        st.session_state.stages: List[List[str]] = []
    if "constraints" not in st.session_state:
        st.session_state.constraints: List[str] = []
    if "last_error" not in st.session_state:
        st.session_state.last_error = ""
    # Persistent Simulation UI state
    if "sim_open" not in st.session_state:
        st.session_state.sim_open = False
    if "sim_cfg" not in st.session_state:
        st.session_state.sim_cfg = {"vary": None, "start": 10.0, "stop": 100.0, "steps": 10, "fixed": {}}
init_state()

def sync_player_order():
    names = list(st.session_state.players.keys())
    order = [n for n in st.session_state.player_order if n in names]
    for n in names:
        if n not in order:
            order.append(n)
    st.session_state.player_order = order
sync_player_order()

# -------------------- Helpers --------------------
def normalize_vars_line(s: str) -> List[str]:
    return [x for x in s.replace(",", " ").split() if x]

def unique_name(base: str) -> str:
    if base not in st.session_state.players:
        return base
    i = 2
    while f"{base}{i}" in st.session_state.players:
        i += 1
    return f"{base}{i}"

def rename_player(old: str, new: str):
    if not new or new == old: return
    if new in st.session_state.players: return
    st.session_state.players[new] = st.session_state.players.pop(old)
    st.session_state.stages = [[(new if x == old else x) for x in stage] for stage in st.session_state.stages]
    st.session_state.player_order = [new if x == old else x for x in st.session_state.player_order]

def infer_parameters_from_utilities(players: Dict[str, Dict]) -> List[str]:
    all_vars: Set[str] = set()
    for pdata in players.values():
        all_vars |= set(pdata.get("vars", []))
    locals_map = {vn: sp.Symbol(vn, real=True) for vn in all_vars}
    params: Set[str] = set()
    for pdata in players.values():
        util_txt = (pdata.get("utility", "") or "").strip()
        if not util_txt: continue
        try:
            expr = sp.sympify(util_txt, locals=locals_map)
            for s in expr.free_symbols:
                if s.name not in all_vars: params.add(s.name)
        except Exception:
            pass
    return sorted(params)

def collect_all_variables(players: Dict[str, Dict]) -> List[str]:
    out: Set[str] = set()
    for pdata in players.values():
        out |= set(pdata.get("vars", []))
    return sorted(out)

def build_spec() -> Dict:
    players_list = []
    for name in st.session_state.player_order:
        pdata = st.session_state.players[name]
        if not name.strip(): continue
        utility = (pdata.get("utility", "") or "").strip()
        if not utility: continue
        players_list.append({"name":name, "vars":pdata.get("vars", []), "utility":utility})
    inferred_params = [{"name": p} for p in infer_parameters_from_utilities(st.session_state.players)]
    constraints = [{"kind":"raw", "expr":c.strip()} for c in st.session_state.constraints if c.strip()]
    if st.session_state.sequential:
        stages = [{"players": s} for s in st.session_state.stages if s]
    else:
        stages = [{"players":[p["name"] for p in players_list]}] if players_list else []
    return {"players":players_list, "parameters":inferred_params, "constraints":constraints, "stages":stages}

def current_param_names() -> List[str]:
    return infer_parameters_from_utilities(st.session_state.players)

# ---------- Parsing constraints (used in simulation filtering only) ----------
OPS = [">=", "<=", ">", "<", "="]

def parse_constraint(raw: str, symbols_map: Dict[str, sp.Symbol]) -> Optional[Tuple[sp.Expr, str, sp.Expr]]:
    s = (raw or "").strip()
    if not s: return None
    op_found = None
    for op in OPS:
        if op in s: op_found = op; break
    if not op_found:
        try:
            lhs = sp.sympify(s, locals=symbols_map); rhs = sp.Integer(0); return lhs, "=", rhs
        except Exception:
            return None
    parts = s.split(op_found)
    if len(parts) != 2: return None
    left, right = parts[0].strip(), parts[1].strip()
    try:
        lhs = sp.sympify(left, locals=symbols_map)
        rhs = sp.sympify(right, locals=symbols_map)
        return lhs, op_found, rhs
    except Exception:
        return None

# -------------------- Sidebar: Reset / Play order / Constraints --------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    if st.button("Reset model"):
        st.session_state.players = {"A":{"vars_line":"q1","vars":["q1"],"utility":""}}
        st.session_state.player_order = ["A"]
        st.session_state.sequential = False
        st.session_state.stages = []
        st.session_state.constraints = []
        st.session_state.sim_open = False
        st.session_state.sim_cfg = {"vary": None, "start": 10.0, "stop": 100.0, "steps": 10, "fixed": {}}
        st.rerun()

    st.markdown("<div class='sidebar-card'></div>", unsafe_allow_html=True)

    st.markdown("#### Play order")
    st.session_state.sequential = st.radio(
        "Mode", options=["Simultaneous","Sequential"],
        index=1 if st.session_state.sequential else 0, horizontal=True, label_visibility="collapsed"
    ) == "Sequential"

    if st.session_state.sequential:
        if not st.session_state.stages:
            init_players = list(st.session_state.player_order)[:1]
            st.session_state.stages = [init_players]
        for si, stage in enumerate(list(st.session_state.stages)):
            st.caption(f"Stage {si+1}")
            options = list(st.session_state.player_order)
            selected = st.multiselect("Players", options, default=stage, key=f"stage_{si}", label_visibility="collapsed")
            st.session_state.stages[si] = selected
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("Add stage", key=f"add_stage_{si}"):
                    st.session_state.stages.insert(si+1, []); st.rerun()
            with sc2:
                if st.button("Remove stage", key=f"del_stage_{si}"):
                    st.session_state.stages.pop(si); st.rerun()

    st.markdown("<div class='sidebar-card'></div>", unsafe_allow_html=True)

    st.markdown("#### Constraints")
    st.caption("Write equations/inequalities (e.g., `q1 >= 0`, `R = a*q1 + b*q2`).")
    for i, c in enumerate(list(st.session_state.constraints)):
        cc1, cc2 = st.columns([4,1])
        with cc1:
            newc = st.text_input("Constraint", value=c, key=f"constraint_{i}",
                                 label_visibility="collapsed",
                                 placeholder="e.g., q1 >= 0  or  R = a*q1 + b*q2")
            if newc != c:
                st.session_state.constraints[i] = newc
        with cc2:
            if st.button("🗑️", key=f"del_constraint_{i}", help="Delete"):
                st.session_state.constraints.pop(i); st.rerun()
    if st.button("Add constraint"):
        st.session_state.constraints.append(""); st.rerun()

# -------------------- Main layout --------------------
left, right = st.columns([1.1, 0.9])

# ============================================================
# Left panel — Players + actions + detected parameters
# ============================================================
with left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Players</div>", unsafe_allow_html=True)

    # Players list (titles: Player 1/2/... + editable Name)
    for idx, pname in enumerate(list(st.session_state.player_order)):
        pdata = st.session_state.players[pname]
        st.markdown("<div class='player-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='player-title'>Player {idx+1}</div>", unsafe_allow_html=True)

        c_name, c_vars, c_actions = st.columns([3,3,1])
        with c_name:
            new_name = st.text_input("Name", value=pname, key=f"name_{pname}",
                                     placeholder="e.g., A", label_visibility="visible")
        with c_vars:
            vars_line = st.text_input("Variables", value=pdata.get("vars_line",""),
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

        pdata["utility"] = st.text_area("Utility function", value=pdata.get("utility",""),
                                        key=f"util_{pname}", height=100,
                                        placeholder="e.g., (a - b*(q1+q2) - c1)*q1",
                                        label_visibility="visible")

        st.markdown("</div>", unsafe_allow_html=True)  # /player-card

    # Add player (bottom)
    if st.button("➕ Add player", key="add_player_btn_bottom", use_container_width=True):
        # Add J2, J3, …
        base = "J"
        i = 2
        while f"{base}{i}" in st.session_state.players:
            i += 1
        newn = f"{base}{i}"
        st.session_state.players[newn] = {"vars_line":"", "vars": [], "utility": ""}
        st.session_state.player_order.append(newn)
        st.rerun()

    # Primary actions
    act_l, act_r = st.columns(2)
    with act_l:
        solve_clicked = st.button("⚙️ Solve ", key="solve_btn", use_container_width=True)
    with act_r:
        sim_clicked   = st.button("📈 Simulation (numeric)", key="sim_btn", use_container_width=True)
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
# Right panel — Results (placeholder when empty)
# ============================================================
with right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    showed_anything = False

    # ---------------- Solve (modified formatting) ----------------
    if 'solve_clicked' in locals() and solve_clicked:
        showed_anything = True
        try:
            spec = build_spec(); gm = build_model(spec); st.session_state.last_error = ""

            var_names = collect_all_variables(st.session_state.players)
            par_names = [p["name"] for p in spec["parameters"]]
            sym_locals = {n: sp.Symbol(n, real=True) for n in (var_names + par_names)}

            # Helper: group variable solutions by player name
            def group_solutions_by_player(sol: Dict[sp.Symbol, sp.Expr]) -> Dict[str, List[Tuple[str, sp.Expr]]]:
                # Build var->player map from current UI spec
                var_to_player: Dict[str, str] = {}
                for pl in spec["players"]:
                    for v in pl.get("vars", []):
                        var_to_player[v] = pl["name"]
                grouped: Dict[str, List[Tuple[str, sp.Expr]]] = {pl["name"]: [] for pl in spec["players"]}
                for k, v in sol.items():
                    vname = str(k) if isinstance(k, sp.Symbol) else str(k)
                    owner = var_to_player.get(vname, "Other")
                    grouped.setdefault(owner, []).append((vname, v))
                # Stable order by players as in spec
                ordered: Dict[str, List[Tuple[str, sp.Expr]]] = {}
                for pl in spec["players"]:
                    ordered[pl["name"]] = grouped.get(pl["name"], [])
                # Append “Other” if anything unmatched
                other_left = [item for name, items in grouped.items() if name not in ordered for item in items]
                if other_left:
                    ordered["Other"] = other_left
                return ordered

            if len(gm.stages) == 1:
                sol, soc = solve_stage_symbolic(gm, gm.stages[0])
                st.subheader("Solve — Simultaneous")

                # 1) Solutions block (by player)
                grouped = group_solutions_by_player(sol)
                lines = []
                for player_name, items in grouped.items():
                    lines.append(f"{player_name}:")
                    if items:
                        for (vname, expr) in items:
                            lines.append(f"  {vname} = {expr}")
                    else:
                        lines.append("  —")
                    lines.append("")
                st.markdown("<div class='result'><h4>Solutions</h4><pre>" + "\n".join(lines) + "</pre></div>", unsafe_allow_html=True)

                # 2) Second-order derivatives (SOC) block (kept)
                soc_lines = []
                for player, values in soc.items():
                    soc_lines.append(f"{player}:")
                    for var, expr in values.items():
                        soc_lines.append(f"  d²U/d{var}² -> {expr}")
                    soc_lines.append("")
                st.markdown("<div class='result'><h4>Second derivatives (should be negative at optimum)</h4><pre>" + "\n".join(soc_lines) + "</pre></div>", unsafe_allow_html=True)

                # NOTE: Constraints block intentionally removed from results (per request)

            else:
                out = solve_by_backward_induction_symbolic(gm)
                st.subheader("Solve — Sequential")

                # 1) Solutions block by player (variables)
                # out["variables"] is a dict var -> expr, we need to group by owner
                var_to_player: Dict[str, str] = {}
                for pl in spec["players"]:
                    for v in pl.get("vars", []):
                        var_to_player[v] = pl["name"]
                grouped: Dict[str, List[Tuple[str, sp.Expr]]] = {pl["name"]: [] for pl in spec["players"]}
                for k, v in out["variables"].items():
                    owner = var_to_player.get(str(k), "Other")
                    grouped.setdefault(owner, []).append((str(k), v))
                lines = []
                for pl in spec["players"]:
                    name = pl["name"]
                    lines.append(f"{name}:")
                    items = grouped.get(name, [])
                    if items:
                        for (vn, ex) in items:
                            lines.append(f"  {vn} = {ex}")
                    else:
                        lines.append("  —")
                    lines.append("")
                st.markdown("<div class='result'><h4>Solutions</h4><pre>" + "\n".join(lines) + "</pre></div>", unsafe_allow_html=True)

                # 2) Utilities block (kept)
                util_text = []
                for k, v in out["utilities"].items():
                    util_text.append(f"{k} = {v}")
                st.markdown("<div class='result'><h4>Utilities</h4><pre>" + "\n".join(util_text) + "</pre></div>", unsafe_allow_html=True)

                # NOTE: Constraints block intentionally removed

        except ModelValidationError as e:
            st.session_state.last_error = str(e); st.error(st.session_state.last_error)
        except SymSolveError as e:
            st.session_state.last_error = str(e); st.warning(st.session_state.last_error)
        except Exception as e:
            st.session_state.last_error = str(e); st.exception(e)

    # ---------------- Simulation (persistent with sim_open + form) ----------------
    if st.session_state.sim_open:
        showed_anything = True
        try:
            spec = build_spec(); gm = build_model(spec)
            param_names = [p["name"] for p in spec["parameters"]]

            col_header_l, col_header_r = st.columns([3,1])
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
                        cfg["stop"]  = st.number_input("End", value=float(cfg["stop"]))
                    cfg["steps"]   = st.number_input("Steps", value=int(cfg["steps"]), min_value=2, step=1)

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
                            if parsed_item: parsed.append(parsed_item)

                        def row_satisfies(row: pd.Series) -> bool:
                            subs_num = {}
                            for v in var_names:
                                if v in row.index: subs_num[all_syms[v]] = float(row[v])
                            subs_num[all_syms[cfg["vary"]]] = float(row["param"])
                            for k, val in fixed_vals.items(): subs_num[all_syms[k]] = float(val)
                            for (lhs, op, rhs) in parsed:
                                l = float(sp.N(lhs.subs(subs_num))); r = float(sp.N(rhs.subs(subs_num)))
                                if op == "=" and abs(l - r) > 1e-8: return False
                                if op == ">=" and not (l >= r - 1e-12): return False
                                if op == "<=" and not (l <= r + 1e-12): return False
                                if op == ">"  and not (l >  r + 1e-12): return False
                                if op == "<"  and not (l <  r - 1e-12): return False
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
        st.markdown("<div class='placeholder'>Results… Solve analytically or run a numeric simulation.</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
