import io
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize

st.set_page_config(page_title="КАДВИ: АПК Оптимизации Запасов", layout="wide")
st.title("🛠 АПК управления запасами ПАО «КАДВИ»")
st.markdown("### Итоговая экономико-математическая модель (Дипломный проект)")

# --- БАЗА ДАННЫХ И ИСХОДНАЯ СТРУКТУРА ---
DB_NAME = Path(__file__).with_name("kadvi_model.db")

SCHEMA_COLUMNS = [
    "id", "name", "C", "H", "S", "T", "P", "R", "K", "D", "D_fuzzy_min", "D_fuzzy_max",
    "L", "L_fuzzy_min", "L_fuzzy_max", "W_i", "N_max", "B_max", "Q_min", "Q_max",
    "SS_min", "SS_max", "I_min", "V", "E", "Z", "M_max", "d_dist", "T_max", "A",
    "G", "Fi_cost", "U_max", "Y_prod", "Y_min", "C_max",
]

NUMERIC_COLUMNS = [c for c in SCHEMA_COLUMNS if c not in {"id", "name"}]

COLUMN_SQL_TYPES = {
    "name": "TEXT", "C": "REAL", "H": "REAL", "S": "REAL", "T": "REAL", "P": "REAL",
    "R": "REAL", "K": "REAL", "D": "REAL", "D_fuzzy_min": "REAL", "D_fuzzy_max": "REAL",
    "L": "REAL", "L_fuzzy_min": "REAL", "L_fuzzy_max": "REAL", "W_i": "REAL", "N_max": "REAL",
    "B_max": "REAL", "Q_min": "REAL", "Q_max": "REAL", "SS_min": "REAL", "SS_max": "REAL",
    "I_min": "REAL", "V": "REAL", "E": "REAL", "Z": "REAL", "M_max": "REAL", "d_dist": "REAL",
    "T_max": "REAL", "A": "REAL", "G": "REAL", "Fi_cost": "REAL", "U_max": "REAL",
    "Y_prod": "REAL", "Y_min": "REAL", "C_max": "REAL",
}

DEFAULTS = {
    "name": "Новая позиция", "C": 0.0, "H": 0.0, "S": 0.0, "T": 0.0, "P": 0.0, "R": 0.0,
    "K": 0.0, "D": 0.0, "D_fuzzy_min": 0.0, "D_fuzzy_max": 0.0, "L": 0.0, "L_fuzzy_min": 0.0,
    "L_fuzzy_max": 0.0, "W_i": 0.0, "N_max": 9999.0, "B_max": 0.0, "Q_min": 0.0, "Q_max": 0.0,
    "SS_min": 0.0, "SS_max": 0.0, "I_min": 0.0, "V": 0.0, "E": 0.0, "Z": 0.0, "M_max": 9999.0,
    "d_dist": 0.0, "T_max": 0.0, "A": 0.0, "G": 0.0, "Fi_cost": 0.0, "U_max": 1.0,
    "Y_prod": 0.0, "Y_min": 0.0, "C_max": 0.0,
}

TABLE_COLUMN_ORDER = ["name", "Q_max", "I_min", "SS_min", "SS_max", "B_max", "N_max", "M_max", "U_max", "Y_prod", "d_dist"]

def seed_rows():
    rows = [
        {"name": "Вал коленчатый", "C": 2500, "H": 120, "S": 450, "T": 80, "P": 0.15, "R": 0.95, "K": 0.98, "D": 400, "D_fuzzy_min": 350, "D_fuzzy_max": 450, "L": 10, "L_fuzzy_min": 8, "L_fuzzy_max": 14, "W_i": 500, "N_max": 12, "B_max": 40, "Q_min": 20, "Q_max": 600, "SS_min": 15, "SS_max": 120, "I_min": 10, "V": 0.5, "E": 1.2, "Z": 0.8, "M_max": 5, "d_dist": 150, "T_max": 50000, "A": 12, "G": 300, "Fi_cost": 500, "U_max": 0.85, "Y_prod": 100, "Y_min": 50, "C_max": 3000},
        {"name": "Шестерня", "C": 850, "H": 40, "S": 200, "T": 30, "P": 0.10, "R": 0.98, "K": 0.99, "D": 1200, "D_fuzzy_min": 1100, "D_fuzzy_max": 1300, "L": 5, "L_fuzzy_min": 4, "L_fuzzy_max": 7, "W_i": 1000, "N_max": 24, "B_max": 100, "Q_min": 50, "Q_max": 1500, "SS_min": 40, "SS_max": 200, "I_min": 20, "V": 0.3, "E": 0.9, "Z": 0.4, "M_max": 10, "d_dist": 80, "T_max": 45000, "A": 8, "G": 150, "Fi_cost": 200, "U_max": 0.90, "Y_prod": 200, "Y_min": 100, "C_max": 1200},
        {"name": "Корпус", "C": 4200, "H": 300, "S": 800, "T": 250, "P": 0.20, "R": 0.90, "K": 0.95, "D": 150, "D_fuzzy_min": 130, "D_fuzzy_max": 180, "L": 20, "L_fuzzy_min": 15, "L_fuzzy_max": 30, "W_i": 300, "N_max": 6, "B_max": 20, "Q_min": 5, "Q_max": 250, "SS_min": 10, "SS_max": 80, "I_min": 5, "V": 0.7, "E": 1.5, "Z": 1.2, "M_max": 3, "d_dist": 300, "T_max": 40000, "A": 20, "G": 600, "Fi_cost": 800, "U_max": 0.80, "Y_prod": 50, "Y_min": 20, "C_max": 5000},
    ]
    return pd.DataFrame(rows)

def ensure_schema(conn: sqlite3.Connection) -> None:
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT, C REAL, H REAL, S REAL, T REAL, P REAL, R REAL, K REAL,
        D REAL, D_fuzzy_min REAL, D_fuzzy_max REAL, L REAL, L_fuzzy_min REAL, L_fuzzy_max REAL,
        W_i REAL, N_max REAL, B_max REAL, Q_min REAL, Q_max REAL, SS_min REAL, SS_max REAL, I_min REAL,
        V REAL, E REAL, Z REAL, M_max REAL, d_dist REAL, T_max REAL, A REAL, G REAL, Fi_cost REAL, U_max REAL,
        Y_prod REAL, Y_min REAL, C_max REAL
    )
    """
    conn.execute(create_sql)
    cur = conn.execute("PRAGMA table_info(inventory)")
    existing_cols = {row[1] for row in cur.fetchall()}
    for col in SCHEMA_COLUMNS:
        if col != "id" and col not in existing_cols:
            conn.execute(f"ALTER TABLE inventory ADD COLUMN {col} {COLUMN_SQL_TYPES[col]}")

def seed_database(force_reset: bool = False) -> None:
    with sqlite3.connect(DB_NAME) as conn:
        if force_reset:
            conn.execute("DROP TABLE IF EXISTS inventory")
            conn.commit()
        ensure_schema(conn)
        cur = conn.execute("SELECT COUNT(*) FROM inventory")
        if cur.fetchone()[0] == 0:
            save_inventory_df(seed_rows(), conn=conn, append=False)

def normalize_text(value, default=""):
    if pd.isna(value): return default
    text = str(value).strip()
    return text if text else default

def row_id_label(row: pd.Series, fallback_index: int) -> str:
    value = row.get("id", pd.NA)
    if pd.isna(value): return f"новая-{fallback_index + 1}"
    return str(int(value))

def coerce_numeric(series: pd.Series, default: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)

def infer_defaults_from_row(row: pd.Series) -> dict:
    d = dict(DEFAULTS)
    c, w_i, q_min, d_det = float(row.get("C", 0) or 0), float(row.get("W_i", 0) or 0), float(row.get("Q_min", 0) or 0), float(row.get("D", 0) or 0)
    l, t, ss_min = float(row.get("L", 0) or 0), float(row.get("T", 0) or 0), float(row.get("SS_min", 0) or 0)
    q_max = float(row.get("Q_max", max(q_min, 1000)) or max(q_min, 1000))

    d["name"] = normalize_text(row.get("name"), "Новая позиция")
    d["C"], d["H"], d["S"], d["T"], d["P"], d["R"], d["K"], d["D"] = c, float(row.get("H", 0) or 0), float(row.get("S", 0) or 0), t, float(row.get("P", 0) or 0), float(row.get("R", 0) or 0), float(row.get("K", 0) or 0), d_det
    d["D_fuzzy_min"], d["D_fuzzy_max"] = float(row.get("D_fuzzy_min", max(0, d_det * 0.9)) or max(0, d_det * 0.9)), float(row.get("D_fuzzy_max", max(d_det, d_det * 1.1)) or max(d_det, d_det * 1.1))
    d["L"], d["L_fuzzy_min"], d["L_fuzzy_max"] = l, float(row.get("L_fuzzy_min", max(0, l * 0.8)) or max(0, l * 0.8)), float(row.get("L_fuzzy_max", max(l, l * 1.2, 1.0)) or max(l, l * 1.2, 1.0))
    d["W_i"], d["N_max"], d["B_max"], d["Q_min"], d["Q_max"] = w_i if w_i > 0 else 1000.0, float(row.get("N_max", 9999) or 9999), float(row.get("B_max", max(0, d_det)) or max(0, d_det)), q_min, q_max if q_max > 0 else max(q_min, 1000.0)
    d["SS_min"], d["SS_max"], d["I_min"], d["V"], d["E"], d["Z"] = ss_min, float(row.get("SS_max", max(ss_min, w_i * 0.8 if w_i > 0 else 1000)) or max(ss_min, w_i * 0.8 if w_i > 0 else 1000)), float(row.get("I_min", 0) or 0), float(row.get("V", 0) or 0), float(row.get("E", 0) or 0), float(row.get("Z", 0) or 0)
    d["M_max"], d["d_dist"], d["T_max"], d["A"], d["G"], d["Fi_cost"] = float(row.get("M_max", 9999) or 9999), float(row.get("d_dist", 0) or 0), float(row.get("T_max", max(1.0, t * max(q_max, q_min, 1.0) * 1.2)) or max(1.0, t * max(q_max, q_min, 1.0) * 1.2)), float(row.get("A", 0) or 0), float(row.get("G", 0) or 0), float(row.get("Fi_cost", 0) or 0)
    d["U_max"], d["Y_prod"], d["Y_min"], d["C_max"] = float(row.get("U_max", 1.0) or 1.0), float(row.get("Y_prod", 0) or 0), float(row.get("Y_min", 0) or 0), float(row.get("C_max", max(c * 1.2, c)) or max(c * 1.2, c))
    return d

def normalize_inventory_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: df = seed_rows()
    else: df = df.copy()

    for col in SCHEMA_COLUMNS:
        if col not in df.columns: df[col] = pd.NA if col == "id" else DEFAULTS[col]

    df["name"] = df["name"].apply(lambda x: normalize_text(x, "Новая позиция"))
    for col in NUMERIC_COLUMNS:
        df[col] = coerce_numeric(df[col], DEFAULTS[col])

    rows = [infer_defaults_from_row(row) for _, row in df.iterrows()]
    normalized = pd.DataFrame(rows)
    normalized.insert(0, "id", pd.to_numeric(df["id"], errors="coerce") if "id" in df.columns else pd.Series([pd.NA] * len(normalized)))
    return normalized[SCHEMA_COLUMNS]

def load_inventory_df() -> pd.DataFrame:
    seed_database()
    with sqlite3.connect(DB_NAME) as conn:
        ensure_schema(conn)
        return normalize_inventory_df(pd.read_sql_query("SELECT * FROM inventory ORDER BY id", conn))

def save_inventory_df(df: pd.DataFrame, conn: sqlite3.Connection | None = None, append: bool = False) -> None:
    df = normalize_inventory_df(df)
    records = df.copy()
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_NAME)
        close_conn = True
    try:
        ensure_schema(conn)
        if not append: conn.execute("DELETE FROM inventory")
        cols = SCHEMA_COLUMNS
        sql = f"INSERT INTO inventory ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"
        valid_ids = pd.to_numeric(records["id"], errors="coerce").dropna().astype(int)
        id_counts = valid_ids.value_counts()
        data = []
        for _, r in records.iterrows():
            row_values = []
            for c in cols:
                if c == "id":
                    if pd.isna(r[c]): row_values.append(None)
                    else:
                        id_value = int(r[c])
                        row_values.append(id_value if id_counts.get(id_value, 0) == 1 else None)
                elif c == "name": row_values.append(normalize_text(r[c], "Новая позиция"))
                else: row_values.append(None if pd.isna(r[c]) else float(r[c]))
            data.append(tuple(row_values))
        conn.executemany(sql, data)
        conn.commit()
    finally:
        if close_conn: conn.close()

# --- МАТЕМАТИКА ОПТИМИЗАТОРА ---
def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b if abs(b) > eps else eps)

def objective_and_metrics(x: np.ndarray, df: pd.DataFrame, mode: str, weights: dict) -> tuple[float, dict, list[dict]]:
    n = len(df)
    metrics = {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0, "F5": 0.0, "F6": 0.0}
    per_item = []

    for i in range(n):
        idx = i * 4
        Q, I, SS, B = float(x[idx]), float(x[idx + 1]), float(x[idx + 2]), float(x[idx + 3])
        r = df.iloc[i]

        D, D_fuzzy, L, L_fuzzy = float(r["D"]), 0.5 * (float(r["D_fuzzy_min"]) + float(r["D_fuzzy_max"])), float(r["L"]), 0.5 * (float(r["L_fuzzy_min"]) + float(r["L_fuzzy_max"]))
        R, K, C, H, S, T, P = max(float(r["R"]), 1e-6), max(float(r["K"]), 0.0), max(float(r["C"]), 0.0), max(float(r["H"]), 0.0), max(float(r["S"]), 0.0), max(float(r["T"]), 0.0), max(float(r["P"]), 0.0)
        V, E, Z, G, Fi_cost, A = max(float(r["V"]), 0.0), max(float(r["E"]), 0.0), max(float(r["Z"]), 0.0), max(float(r["G"]), 0.0), max(float(r["Fi_cost"]), 0.0), max(float(r["A"]), 0.0)
        d_dist, U_max, W_i, Y_prod = max(float(r["d_dist"]), 0.0), float(r["U_max"]), max(float(r["W_i"]), 1e-6), max(float(r["Y_prod"]), 0.0)

        Q_safe = max(Q, 1e-6)
        N_i = safe_div(D, Q_safe)
        M_i = safe_div(D, Q_safe)
        U_i = min(max(I / W_i, 0.0), max(U_max, 0.0))

        F1_i = C * Q + H * I + S * N_i + T * Q + P * B + R * SS
        R0, C0_factor = float(weights.get("R0", 0.95)), float(weights.get("C0_factor", 0.9))
        C0 = max(C * max(float(r["Q_min"]), 1.0) * C0_factor, 1e-6)

        F2_i = (
            weights["alpha"] * (D_fuzzy - Q)**2 +
            weights["beta"] * (L_fuzzy - L)**2 +
            weights["gamma"] * (B**2) +
            weights["delta"] * ((SS - I)**2) +
            weights["lambda"] * ((R - R0)**2 * 1000) +
            weights["theta"] * (((C * Q - C0) / 1000.0)**2)
        )

        F3_i = ((Q + SS - B) * R * K) / max(D_fuzzy + L, 1e-6)
        F4_i = P * L * Q + V * B + E * M_i + (Z / max(R, 1e-6))
        F5_i = A * d_dist + G * M_i + Fi_cost * U_i
        F6_i = (I * U_i * Y_prod) / max(W_i + C + L, 1e-6)

        metrics["F1"] += F1_i; metrics["F2"] += F2_i; metrics["F3"] += F3_i
        metrics["F4"] += F4_i; metrics["F5"] += F5_i; metrics["F6"] += F6_i

        per_item.append({
            "ID": row_id_label(r, i), "Товар": r["name"],
            "Q_i": Q, "I_i": I, "SS_i": SS, "B_i": B, "N_i": N_i, "L_i": float(r["L"]),
            "M_i": M_i, "U_i": U_i, "Y_i": float(r["Y_prod"]), "d_i": float(r["d_dist"]),
            "F1_i": F1_i, "F2_i": F2_i, "F3_i": F3_i, "F4_i": F4_i, "F5_i": F5_i, "F6_i": F6_i,
        })

    if mode.startswith("F1"): total = metrics["F1"]
    elif mode.startswith("F2"): total = metrics["F2"]
    elif mode.startswith("F3"): total = -metrics["F3"]
    elif mode.startswith("F4"): total = metrics["F4"]
    elif mode.startswith("F5"): total = metrics["F5"]
    elif mode.startswith("F6"): total = -metrics["F6"]
    else: total = metrics["F1"]

    return total, metrics, per_item

def validate_inputs(df_input: pd.DataFrame, budget: float, capacity: float) -> list[str]:
    errors = []
    required_numeric = ["C", "H", "S", "T", "P", "R", "K", "D", "L", "W_i", "Q_min", "Q_max", "SS_min", "SS_max"]
    for i, row in df_input.iterrows():
        name = normalize_text(row.get("name"), f"Строка {i + 1}")
        for c in required_numeric:
            if pd.isna(row.get(c)) or not np.isfinite(float(row.get(c))): errors.append(f"🔴 **{name}**: поле `{c}` некорректно.")
        q_max, ss_max, d = float(row["Q_max"]), float(row["SS_max"]), float(row["D"])
        if q_max + ss_max < d: errors.append(f"🔴 **{name}**: Q_max + SS_max = {q_max + ss_max:.2f} не покрывает спрос D = {d:.2f}.")

    min_budget_needed = float((df_input["C"] * df_input["Q_min"]).sum())
    if min_budget_needed > budget: errors.append(f"💰 **Бюджет**: для закупки минимума требуется {min_budget_needed:,.2f} руб., доступно {budget:,.2f} руб.")
    return errors

def build_bounds(df_input: pd.DataFrame) -> tuple[list[tuple[float, float]], list[float]]:
    bounds, x0 = [], []
    for _, row in df_input.iterrows():
        d, q_min, q_max, ss_min, ss_max, i_min, w_i, b_max = float(row["D"]), float(row["Q_min"]), float(row["Q_max"]), float(row["SS_min"]), float(row["SS_max"]), float(row["I_min"]), float(row["W_i"]), float(row["B_max"])
        u_max = max(float(row["U_max"]), 0.0)

        q_lower, q_upper = q_min, max(q_min, q_max)
        i_lower, i_upper = i_min, max(i_min, w_i * u_max)
        ss_lower, ss_upper = ss_min, min(ss_max, w_i)
        if ss_upper < ss_lower: ss_upper = ss_lower
        b_lower, b_upper = 0.0, min(b_max, max(d, 0.0))

        bounds.extend([(q_lower, q_upper), (i_lower, i_upper), (ss_lower, ss_upper), (b_lower, b_upper)])
        x0.extend([(q_lower + q_upper) / 2.0, (i_lower + i_upper) / 2.0, (ss_lower + ss_upper) / 2.0, b_upper / 2.0])
    return bounds, x0

# Глобальная оптимизация (с учетом лимитов бюджета и склада)
def run_optimization(df_input: pd.DataFrame, mode: str, globals_cfg: dict, weights: dict):
    n = len(df_input)
    bounds, x0 = build_bounds(df_input)
    SCALE = 1_000_000.0

    def objective(x: np.ndarray) -> float:
        val, _, _ = objective_and_metrics(x, df_input, mode, weights)
        return float(val) / SCALE

    cons = []
    cons.append({"type": "ineq", "fun": lambda x: (globals_cfg["F_budget"] - float(sum(x[i * 4] * float(df_input.iloc[i]["C"]) for i in range(n)))) / SCALE})
    cons.append({"type": "ineq", "fun": lambda x: (globals_cfg["W_total"] - float(sum(x[i * 4 + 1] for i in range(n)))) / 1000.0})

    for i in range(n):
        def bal_con(x, i=i): return (x[i * 4] + x[i * 4 + 2]) - x[i * 4 + 3] - float(df_input.iloc[i]["D"])
        cons.append({"type": "ineq", "fun": bal_con})

        row = df_input.iloc[i]
        d = float(row["D"])
        n_max = max(float(row["N_max"]), 1e-9)
        m_max = max(float(row["M_max"]), 1e-9)
        t_max = float(row["T_max"])

        def n_con(x, i=i, d=d, n_max=n_max): return n_max - safe_div(d, max(float(x[i * 4]), 1e-9))
        cons.append({"type": "ineq", "fun": n_con})
        def m_con(x, i=i, d=d, m_max=m_max): return m_max - safe_div(d, max(float(x[i * 4]), 1e-9))
        cons.append({"type": "ineq", "fun": m_con})

        if t_max > 0:
            def t_con(x, i=i, t_max=t_max): return (t_max - float(df_input.iloc[i]["T"]) * float(x[i * 4])) / SCALE
            cons.append({"type": "ineq", "fun": t_con})

    return minimize(objective, np.array(x0, dtype=float), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 3000, "ftol": 1e-9, "disp": False})

# Индивидуальная оптимизация идеальных параметров (без общих лимитов)
def run_item_optimization(row: pd.Series, mode: str, weights: dict):
    df_single = pd.DataFrame([row]).reset_index(drop=True)
    bounds, x0 = build_bounds(df_single)
    SCALE = 1_000_000.0

    def objective(x: np.ndarray) -> float:
        val, _, _ = objective_and_metrics(x, df_single, mode, weights)
        return float(val) / SCALE

    cons = []
    def bal_con(x): return (x[0] + x[2]) - x[3] - float(row["D"])
    cons.append({"type": "ineq", "fun": bal_con})

    d = float(row["D"])
    n_max = max(float(row["N_max"]), 1e-9)
    m_max = max(float(row["M_max"]), 1e-9)
    t_max = float(row["T_max"])

    cons.append({"type": "ineq", "fun": lambda x: n_max - safe_div(d, max(float(x[0]), 1e-9))})
    cons.append({"type": "ineq", "fun": lambda x: m_max - safe_div(d, max(float(x[0]), 1e-9))})

    if t_max > 0:
        cons.append({"type": "ineq", "fun": lambda x: (t_max - float(row["T"]) * float(x[0])) / SCALE})

    return minimize(objective, np.array(x0, dtype=float), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 3000, "ftol": 1e-9, "disp": False})


def score_single_candidate(candidate_df: pd.DataFrame, mode: str, globals_cfg: dict, weights: dict) -> float:
    try:
        result = run_optimization(candidate_df, mode, globals_cfg, weights)
        if result is not None and result.success:
            score, _, _ = objective_and_metrics(result.x, candidate_df, mode, weights)
            return float(score)
    except Exception: pass
    _, x0 = build_bounds(candidate_df)
    score, _, _ = objective_and_metrics(np.array(x0, dtype=float), candidate_df, mode, weights)
    return float(score) + 1e18

def select_best_alternatives(df_input: pd.DataFrame, mode: str, globals_cfg: dict, weights: dict) -> tuple[pd.DataFrame, list[str]]:
    df = df_input.copy()
    if df.empty: return df, []

    df["_name_key"] = df["name"].apply(lambda value: normalize_text(value, "Новая позиция").casefold())
    selected_indices, notes = [], []

    for _, group in df.groupby("_name_key", sort=False):
        if len(group) == 1:
            selected_indices.append(group.index[0])
            continue
        scored_candidates = []
        for idx, row in group.iterrows():
            candidate_df = group.loc[[idx], SCHEMA_COLUMNS].reset_index(drop=True)
            scored_candidates.append((score_single_candidate(candidate_df, mode, globals_cfg, weights), idx))

        _, best_idx = min(scored_candidates, key=lambda item: item[0])
        selected_indices.append(best_idx)
        best_row = df.loc[best_idx]
        notes.append(f"{best_row['name']}: выбран ID {row_id_label(best_row, int(best_idx))} из {len(group)} вариантов")

    selected_df = df.loc[selected_indices, SCHEMA_COLUMNS].reset_index(drop=True)
    return selected_df, notes

# --- ИНТЕРФЕЙС ---

df_current = load_inventory_df()

with st.sidebar:
    st.header("⚙️ Режим расчета")
    mode = st.selectbox("🎯 Целевая функция:", [
        "F1: Минимизация совокупных затрат", "F2: Потери от неопределенности",
        "F3: Макс. уровня обеспечения", "F4: Риск логистических сбоев",
        "F5: Транспортно-складские расходы", "F6: Макс. эффективности склада"
    ])

    weights = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "lambda": 0.0, "theta": 0.0, "R0": 0.95, "C0_factor": 0.9}
    if mode.startswith("F2"):
        st.divider()
        with st.expander("Весовые коэффициенты F2", expanded=True):
            weights["alpha"] = st.slider("alpha — спрос", 0.0, 1.0, 0.30, 0.01)
            weights["beta"] = st.slider("beta — сроки", 0.0, 1.0, 0.20, 0.01)
            weights["gamma"] = st.slider("gamma — дефицит", 0.0, 1.0, 0.10, 0.01)
            weights["delta"] = st.slider("delta — страховой запас", 0.0, 1.0, 0.10, 0.01)
            weights["lambda"] = st.slider("lambda — надежность", 0.0, 1.0, 0.20, 0.01)
            weights["theta"] = st.slider("theta — затраты", 0.0, 1.0, 0.10, 0.01)

    st.divider()
    st.markdown("### 📥 Импорт данных")
    uploaded_file = st.file_uploader("Загрузить БД (Excel)", type=["xlsx"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            save_inventory_df(pd.read_excel(uploaded_file))
            st.success("БД обновлена!")
            st.rerun()
        except: st.error("Ошибка при чтении Excel.")

    st.divider()
    st.markdown("### 🌍 Глобальные переменные")
    F_budget = st.number_input("F (Бюджет, руб)", value=5000000.0, step=100000.0)
    W_total = st.number_input("W (Склад, ед)", value=10000.0, step=1000.0)
    globals_cfg = {"F_budget": F_budget, "W_total": W_total}

    st.divider()
    st.markdown("### 🔒 Постоянные параметры")
    selected_idx = st.selectbox(
        "📝 Выбор позиции:", 
        range(len(df_current)), 
        format_func=lambda x: df_current.iloc[x]['name']
    )
    
    # Распределяем постоянные параметры в 2 колонки для компактности в сайдбаре
    c1, c2 = st.columns(2)
    C_i = c1.number_input("C_i", value=float(df_current.at[selected_idx, "C"]))
    H_i = c2.number_input("H_i", value=float(df_current.at[selected_idx, "H"]))
    S_i = c1.number_input("S_i", value=float(df_current.at[selected_idx, "S"]))
    T_i = c2.number_input("T_i", value=float(df_current.at[selected_idx, "T"]))
    P_i = c1.number_input("P_i", value=float(df_current.at[selected_idx, "P"]))
    R_i = c2.number_input("R_i", value=float(df_current.at[selected_idx, "R"]))
    O_i = c1.number_input("O_i (Спрос)", value=float(df_current.at[selected_idx, "D"]))
    W_i_param = c2.number_input("W_i (Склад)", value=float(df_current.at[selected_idx, "W_i"]))
    K_i = c1.number_input("K_i", value=float(df_current.at[selected_idx, "K"]))
    V_i = c2.number_input("V_i", value=float(df_current.at[selected_idx, "V"]))
    E_i = c1.number_input("E_i", value=float(df_current.at[selected_idx, "E"]))
    Z_i = c2.number_input("Z_i (Риск)", value=float(df_current.at[selected_idx, "Z"]))
    A_i = c1.number_input("A_i", value=float(df_current.at[selected_idx, "A"]))
    G_i = c2.number_input("G_i", value=float(df_current.at[selected_idx, "G"]))
    F_i = c1.number_input("F_i", value=float(df_current.at[selected_idx, "Fi_cost"]))
    Q_i = c2.number_input("Q_i", value=float(df_current.at[selected_idx, "Q_min"]))
    L_i = c1.number_input("L_i", value=float(df_current.at[selected_idx, "L"]))

    df_current.loc[selected_idx, ["C", "H", "S", "T", "P", "R", "D", "W_i", "K", "V", "E", "Z", "A", "G", "Fi_cost", "Q_min", "L"]] = [C_i, H_i, S_i, T_i, P_i, R_i, O_i, W_i_param, K_i, V_i, E_i, Z_i, A_i, G_i, F_i, Q_i, L_i]


# Главная рабочая зона - Таблица теперь на всю ширину
st.markdown("#### 🎛️ Переменные параметры (Ограничения)")
edited_table = st.data_editor(
    df_current[TABLE_COLUMN_ORDER],
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
)
for col in TABLE_COLUMN_ORDER: df_current[col] = edited_table[col]

st.write("") 
c_btn1, c_btn2, _ = st.columns(
