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

DB_NAME = "kadvi_model.db"

SCHEMA_COLUMNS = [
    "id",
    "name",
    "C", "H", "S", "T", "P", "R", "K",
    "D", "D_fuzzy_min", "D_fuzzy_max",
    "L", "L_fuzzy_min", "L_fuzzy_max",
    "W_i", "N_max", "B_max",
    "Q_min", "Q_max", "SS_min", "SS_max", "I_min",
    "V", "E", "Z", "M_max", "d_dist", "T_max",
    "A", "G", "Fi_cost", "U_max",
    "Y_prod", "Y_min", "C_max",
]

NUMERIC_COLUMNS = [c for c in SCHEMA_COLUMNS if c not in {"id", "name"}]

COLUMN_SQL_TYPES = {
    "name": "TEXT",
    "C": "REAL", "H": "REAL", "S": "REAL", "T": "REAL", "P": "REAL", "R": "REAL", "K": "REAL",
    "D": "REAL", "D_fuzzy_min": "REAL", "D_fuzzy_max": "REAL",
    "L": "REAL", "L_fuzzy_min": "REAL", "L_fuzzy_max": "REAL",
    "W_i": "REAL", "N_max": "REAL", "B_max": "REAL",
    "Q_min": "REAL", "Q_max": "REAL", "SS_min": "REAL", "SS_max": "REAL", "I_min": "REAL",
    "V": "REAL", "E": "REAL", "Z": "REAL", "M_max": "REAL", "d_dist": "REAL", "T_max": "REAL",
    "A": "REAL", "G": "REAL", "Fi_cost": "REAL", "U_max": "REAL",
    "Y_prod": "REAL", "Y_min": "REAL", "C_max": "REAL",
}

DEFAULTS = {
    "name": "Новая позиция",
    "C": 0.0, "H": 0.0, "S": 0.0, "T": 0.0, "P": 0.0, "R": 0.0, "K": 0.0,
    "D": 0.0, "D_fuzzy_min": 0.0, "D_fuzzy_max": 0.0,
    "L": 0.0, "L_fuzzy_min": 0.0, "L_fuzzy_max": 0.0,
    "W_i": 0.0, "N_max": 9999.0, "B_max": 0.0,
    "Q_min": 0.0, "Q_max": 0.0, "SS_min": 0.0, "SS_max": 0.0, "I_min": 0.0,
    "V": 0.0, "E": 0.0, "Z": 0.0, "M_max": 9999.0, "d_dist": 0.0, "T_max": 0.0,
    "A": 0.0, "G": 0.0, "Fi_cost": 0.0, "U_max": 1.0,
    "Y_prod": 0.0, "Y_min": 0.0, "C_max": 0.0,
}

def seed_rows():
    rows = [
        {
            "name": "Вал коленчатый",
            "C": 2500, "H": 120, "S": 450, "T": 80, "P": 0.15, "R": 0.95, "K": 0.98,
            "D": 400, "D_fuzzy_min": 350, "D_fuzzy_max": 450,
            "L": 10, "L_fuzzy_min": 8, "L_fuzzy_max": 14,
            "W_i": 500, "N_max": 12, "B_max": 40,
            "Q_min": 20, "Q_max": 600, "SS_min": 15, "SS_max": 120, "I_min": 10,
            "V": 0.5, "E": 1.2, "Z": 0.8, "M_max": 5, "d_dist": 150, "T_max": 50000,
            "A": 12, "G": 300, "Fi_cost": 500, "U_max": 0.85,
            "Y_prod": 100, "Y_min": 50, "C_max": 3000,
        },
        {
            "name": "Шестерня",
            "C": 850, "H": 40, "S": 200, "T": 30, "P": 0.10, "R": 0.98, "K": 0.99,
            "D": 1200, "D_fuzzy_min": 1100, "D_fuzzy_max": 1300,
            "L": 5, "L_fuzzy_min": 4, "L_fuzzy_max": 7,
            "W_i": 1000, "N_max": 24, "B_max": 100,
            "Q_min": 50, "Q_max": 1500, "SS_min": 40, "SS_max": 200, "I_min": 20,
            "V": 0.3, "E": 0.9, "Z": 0.4, "M_max": 10, "d_dist": 80, "T_max": 45000,
            "A": 8, "G": 150, "Fi_cost": 200, "U_max": 0.90,
            "Y_prod": 200, "Y_min": 100, "C_max": 1200,
        },
        {
            "name": "Корпус",
            "C": 4200, "H": 300, "S": 800, "T": 250, "P": 0.20, "R": 0.90, "K": 0.95,
            "D": 150, "D_fuzzy_min": 130, "D_fuzzy_max": 180,
            "L": 20, "L_fuzzy_min": 15, "L_fuzzy_max": 30,
            "W_i": 300, "N_max": 6, "B_max": 20,
            "Q_min": 5, "Q_max": 250, "SS_min": 10, "SS_max": 80, "I_min": 5,
            "V": 0.7, "E": 1.5, "Z": 1.2, "M_max": 3, "d_dist": 300, "T_max": 40000,
            "A": 20, "G": 600, "Fi_cost": 800, "U_max": 0.80,
            "Y_prod": 50, "Y_min": 20, "C_max": 5000,
        },
    ]
    return pd.DataFrame(rows)

def ensure_schema(conn: sqlite3.Connection) -> None:
    create_sql = """
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        C REAL, H REAL, S REAL, T REAL, P REAL, R REAL, K REAL,
        D REAL, D_fuzzy_min REAL, D_fuzzy_max REAL,
        L REAL, L_fuzzy_min REAL, L_fuzzy_max REAL,
        W_i REAL, N_max REAL, B_max REAL,
        Q_min REAL, Q_max REAL, SS_min REAL, SS_max REAL, I_min REAL,
        V REAL, E REAL, Z REAL, M_max REAL, d_dist REAL, T_max REAL,
        A REAL, G REAL, Fi_cost REAL, U_max REAL,
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
    # Открываем соединение и сбрасываем таблицу внутри БД, 
    # вместо удаления самого файла, чтобы избежать WinError 32
    with sqlite3.connect(DB_NAME) as conn:
        if force_reset:
            conn.execute("DROP TABLE IF EXISTS inventory")
            conn.commit()
            
        ensure_schema(conn)
        cur = conn.execute("SELECT COUNT(*) FROM inventory")
        count = cur.fetchone()[0]
        if count == 0:
            seed_df = seed_rows()
            save_inventory_df(seed_df, conn=conn, append=False)
    with sqlite3.connect(DB_NAME) as conn:
        ensure_schema(conn)
        cur = conn.execute("SELECT COUNT(*) FROM inventory")
        count = cur.fetchone()[0]
        if count == 0:
            seed_df = seed_rows()
            save_inventory_df(seed_df, conn=conn, append=False)

def normalize_text(value, default=""):
    if pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default

def coerce_numeric(series: pd.Series, default: float) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    return out.fillna(default).astype(float)

def infer_defaults_from_row(row: pd.Series) -> dict:
    d = dict(DEFAULTS)
    c = float(row.get("C", 0) or 0)
    w_i = float(row.get("W_i", 0) or 0)
    q_min = float(row.get("Q_min", 0) or 0)
    q_max = float(row.get("Q_max", max(q_min, 1000)) or max(q_min, 1000))
    ss_min = float(row.get("SS_min", 0) or 0)
    i_min = float(row.get("I_min", 0) or 0)
    d_det = float(row.get("D", 0) or 0)
    l = float(row.get("L", 0) or 0)
    t = float(row.get("T", 0) or 0)

    d["name"] = normalize_text(row.get("name"), "Новая позиция")
    d["C"] = c
    d["H"] = float(row.get("H", 0) or 0)
    d["S"] = float(row.get("S", 0) or 0)
    d["T"] = t
    d["P"] = float(row.get("P", 0) or 0)
    d["R"] = float(row.get("R", 0) or 0)
    d["K"] = float(row.get("K", 0) or 0)
    d["D"] = d_det
    d["D_fuzzy_min"] = float(row.get("D_fuzzy_min", max(0, d_det * 0.9)) or max(0, d_det * 0.9))
    d["D_fuzzy_max"] = float(row.get("D_fuzzy_max", max(d_det, d_det * 1.1)) or max(d_det, d_det * 1.1))
    d["L"] = l
    d["L_fuzzy_min"] = float(row.get("L_fuzzy_min", max(0, l * 0.8)) or max(0, l * 0.8))
    d["L_fuzzy_max"] = float(row.get("L_fuzzy_max", max(l, l * 1.2, 1.0)) or max(l, l * 1.2, 1.0))
    d["W_i"] = w_i if w_i > 0 else 1000.0
    d["N_max"] = float(row.get("N_max", 9999) or 9999)
    d["B_max"] = float(row.get("B_max", max(0, d_det)) or max(0, d_det))
    d["Q_min"] = q_min
    d["Q_max"] = q_max if q_max > 0 else max(q_min, 1000.0)
    d["SS_min"] = ss_min
    d["SS_max"] = float(row.get("SS_max", max(ss_min, w_i * 0.8 if w_i > 0 else 1000)) or max(ss_min, w_i * 0.8 if w_i > 0 else 1000))
    d["I_min"] = i_min
    d["V"] = float(row.get("V", 0) or 0)
    d["E"] = float(row.get("E", 0) or 0)
    d["Z"] = float(row.get("Z", 0) or 0)
    d["M_max"] = float(row.get("M_max", 9999) or 9999)
    d["d_dist"] = float(row.get("d_dist", 0) or 0)
    d["T_max"] = float(row.get("T_max", max(1.0, t * max(q_max, q_min, 1.0) * 1.2)) or max(1.0, t * max(q_max, q_min, 1.0) * 1.2))
    d["A"] = float(row.get("A", 0) or 0)
    d["G"] = float(row.get("G", 0) or 0)
    d["Fi_cost"] = float(row.get("Fi_cost", 0) or 0)
    d["U_max"] = float(row.get("U_max", 1.0) or 1.0)
    d["Y_prod"] = float(row.get("Y_prod", 0) or 0)
    d["Y_min"] = float(row.get("Y_min", 0) or 0)
    d["C_max"] = float(row.get("C_max", max(c * 1.2, c)) or max(c * 1.2, c))
    return d

def normalize_inventory_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        df = seed_rows()
    else:
        df = df.copy()

    for col in SCHEMA_COLUMNS:
        if col not in df.columns:
            if col == "id":
                df[col] = pd.NA
            else:
                df[col] = DEFAULTS[col]

    df["name"] = df["name"].apply(lambda x: normalize_text(x, "Новая позиция"))

    for col in NUMERIC_COLUMNS:
        df[col] = coerce_numeric(df[col], DEFAULTS[col])

    rows = []
    for _, row in df.iterrows():
        rows.append(infer_defaults_from_row(row))

    normalized = pd.DataFrame(rows)
    if "id" in df.columns:
        ids = pd.to_numeric(df["id"], errors="coerce")
        normalized.insert(0, "id", ids)
    else:
        normalized.insert(0, "id", pd.Series([pd.NA] * len(normalized)))

    normalized = normalized[SCHEMA_COLUMNS]
    return normalized

def load_inventory_df() -> pd.DataFrame:
    seed_database(force_reset=False)
    with sqlite3.connect(DB_NAME) as conn:
        ensure_schema(conn)
        df = pd.read_sql_query("SELECT * FROM inventory ORDER BY id", conn)
    return normalize_inventory_df(df)

def save_inventory_df(df: pd.DataFrame, conn: sqlite3.Connection | None = None, append: bool = False) -> None:
    df = normalize_inventory_df(df)
    records = df.drop(columns=["id"]).copy()

    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DB_NAME)
        close_conn = True

    try:
        ensure_schema(conn)
        if not append:
            conn.execute("DELETE FROM inventory")

        cols = [c for c in SCHEMA_COLUMNS if c != "id"]
        placeholders = ", ".join(["?"] * len(cols))
        sql = f"INSERT INTO inventory ({', '.join(cols)}) VALUES ({placeholders})"

        data = []
        for _, row in records.iterrows():
            values = []
            for c in cols:
                v = row[c]
                if c == "name":
                    values.append(normalize_text(v, "Новая позиция"))
                else:
                    values.append(None if pd.isna(v) else float(v))
            data.append(tuple(values))

        conn.executemany(sql, data)
        conn.commit()
    finally:
        if close_conn:
            conn.close()

def safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return a / (b if abs(b) > eps else eps)

def objective_and_metrics(x: np.ndarray, df: pd.DataFrame, mode: str, weights: dict) -> tuple[float, dict, list[dict]]:
    n = len(df)
    total = 0.0
    metrics = {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0, "F5": 0.0, "F6": 0.0}
    per_item = []

    for i in range(n):
        idx = i * 4
        Q = float(x[idx])
        I = float(x[idx + 1])
        SS = float(x[idx + 2])
        B = float(x[idx + 3])
        r = df.iloc[i]

        D = float(r["D"])
        D_fuzzy = 0.5 * (float(r["D_fuzzy_min"]) + float(r["D_fuzzy_max"]))
        L = float(r["L"])
        L_fuzzy = 0.5 * (float(r["L_fuzzy_min"]) + float(r["L_fuzzy_max"]))
        R = max(float(r["R"]), 1e-6)
        K = max(float(r["K"]), 0.0)
        C = max(float(r["C"]), 0.0)
        H = max(float(r["H"]), 0.0)
        S = max(float(r["S"]), 0.0)
        T = max(float(r["T"]), 0.0)
        P = max(float(r["P"]), 0.0)
        V = max(float(r["V"]), 0.0)
        E = max(float(r["E"]), 0.0)
        Z = max(float(r["Z"]), 0.0)
        G = max(float(r["G"]), 0.0)
        Fi_cost = max(float(r["Fi_cost"]), 0.0)
        A = max(float(r["A"]), 0.0)
        d_dist = max(float(r["d_dist"]), 0.0)
        U_max = float(r["U_max"])
        W_i = max(float(r["W_i"]), 1e-6)
        Y_prod = max(float(r["Y_prod"]), 0.0)

        Q_safe = max(Q, 1e-6)
        N_i = safe_div(D, Q_safe)
        M_i = safe_div(D, Q_safe)
        U_i = min(max(I / W_i, 0.0), max(U_max, 0.0))

        F1_i = C * Q + H * I + S * N_i + T * Q + P * B + R * SS

        R0 = float(weights.get("R0", 0.95))
        C0_factor = float(weights.get("C0_factor", 0.9))
        C0 = max(C * max(float(r["Q_min"]), 1.0) * C0_factor, 1e-6)

        F2_i = (
            weights["alpha"] * (D_fuzzy - Q) ** 2
            + weights["beta"] * (L_fuzzy - L) ** 2
            + weights["gamma"] * (B ** 2)
            + weights["delta"] * ((SS - I) ** 2)
            + weights["lambda"] * ((R - R0) ** 2)
            + weights["theta"] * (((C * Q) - C0) ** 2)
        )

        denom_f3 = max(D_fuzzy + L, 1e-6)
        F3_i = ((Q + SS - B) * R * K) / denom_f3
        F4_i = P * L * Q + V * B + E * M_i + (Z / max(R, 1e-6))
        F5_i = A * d_dist + G * M_i + Fi_cost * U_i
        F6_i = (I * U_i * Y_prod) / max(W_i + C + L, 1e-6)

        metrics["F1"] += F1_i
        metrics["F2"] += F2_i
        metrics["F3"] += F3_i
        metrics["F4"] += F4_i
        metrics["F5"] += F5_i
        metrics["F6"] += F6_i

        per_item.append({
            "Товар": r["name"],
            "Q": Q,
            "I": I,
            "SS": SS,
            "B": B,
            "N_i": N_i,
            "M_i": M_i,
            "U_i": U_i,
            "F1_i": F1_i,
            "F2_i": F2_i,
            "F3_i": F3_i,
            "F4_i": F4_i,
            "F5_i": F5_i,
            "F6_i": F6_i,
        })

    if mode.startswith("F1"):
        total = metrics["F1"]
    elif mode.startswith("F2"):
        total = metrics["F2"]
    elif mode.startswith("F3"):
        total = -metrics["F3"]
    elif mode.startswith("F4"):
        total = metrics["F4"]
    elif mode.startswith("F5"):
        total = metrics["F5"]
    elif mode.startswith("F6"):
        total = -metrics["F6"]
    else:
        total = metrics["F1"]

    return total, metrics, per_item

def validate_inputs(df_input: pd.DataFrame, budget: float, capacity: float, globals_cfg: dict) -> list[str]:
    errors = []
    warnings = []

    required_numeric = [
        "C", "H", "S", "T", "P", "R", "K", "D", "L", "W_i",
        "N_max", "B_max", "Q_min", "Q_max", "SS_min", "SS_max", "I_min",
        "V", "E", "Z", "M_max", "d_dist", "T_max", "A", "G", "Fi_cost",
        "U_max", "Y_prod", "Y_min", "C_max",
    ]

    for i, row in df_input.iterrows():
        name = normalize_text(row.get("name"), f"Строка {i + 1}")

        for c in required_numeric:
            v = row.get(c)
            if pd.isna(v) or not np.isfinite(float(v)):
                errors.append(f"🔴 **{name}**: поле `{c}` содержит некорректное значение.")
                continue
            if float(v) < 0:
                errors.append(f"🔴 **{name}**: поле `{c}` не может быть отрицательным.")

        c_val = float(row["C"])
        d = float(row["D"])
        q_min = float(row["Q_min"])
        q_max = float(row["Q_max"])
        ss_min = float(row["SS_min"])
        ss_max = float(row["SS_max"])
        i_min = float(row["I_min"])
        w_i = float(row["W_i"])
        n_max = float(row["N_max"])
        m_max = float(row["M_max"])
        t = float(row["T"])
        t_max = float(row["T_max"])
        b_max = float(row["B_max"])

        if q_min > q_max:
            errors.append(f"🔴 **{name}**: Q_min ({q_min}) больше Q_max ({q_max}).")
        if ss_min > ss_max:
            errors.append(f"🔴 **{name}**: SS_min ({ss_min}) больше SS_max ({ss_max}).")
        if i_min > w_i:
            errors.append(f"🔴 **{name}**: I_min ({i_min}) больше W_i ({w_i}).")
        if b_max > d and d >= 0:
            warnings.append(f"🟡 **{name}**: B_max ({b_max}) больше спроса D ({d}). Это допустимо, но редко необходимо.")

        if q_max + ss_max < d:
            errors.append(
                f"🔴 **{name}**: даже при Q_max + SS_max = {q_max + ss_max:.2f} "
                f"нельзя покрыть спрос D = {d:.2f}."
            )

        if n_max > 0 and d > 0:
            if q_max < d / n_max:
                errors.append(
                    f"🔴 **{name}**: для ограничения N_max = {n_max:.2f} требуется Q ≥ {d / n_max:.2f}, "
                    f"но Q_max = {q_max:.2f}."
                )
        if m_max > 0 and d > 0:
            if q_max < d / m_max:
                errors.append(
                    f"🔴 **{name}**: для ограничения M_max = {m_max:.2f} требуется Q ≥ {d / m_max:.2f}, "
                    f"но Q_max = {q_max:.2f}."
                )

        if t > 0 and t_max > 0 and (t * q_min) > t_max:
            errors.append(
                f"🔴 **{name}**: даже минимальный заказ Q_min={q_min:.2f} нарушает транспортный лимит "
                f"T_max={t_max:.2f} при T={t:.2f}."
            )

        if float(row["C"]) > float(row["C_max"]):
            errors.append(f"🔴 **{name}**: C ({row['C']:.2f}) больше C_max ({row['C_max']:.2f}).")
        if float(row["Y_prod"]) < float(row["Y_min"]):
            errors.append(f"🔴 **{name}**: Y_prod ({row['Y_prod']:.2f}) меньше Y_min ({row['Y_min']:.2f}).")

    q_min_sum = float(df_input["Q_min"].sum())
    ss_min_sum = float(df_input["SS_min"].sum())
    i_min_sum = float(df_input["I_min"].sum())
    c_sum = float(df_input["C"].sum())
    y_sum = float(df_input["Y_prod"].sum())
    p_l_sum = float((df_input["P"] * df_input["L"]).sum())

    if q_min_sum > globals_cfg["Q_total_max"]:
        errors.append(
            f"💰 **Q^max**: суммарный минимальный объём заказа {q_min_sum:,.2f} превышает лимит {globals_cfg['Q_total_max']:,.2f}."
        )
    if ss_min_sum > globals_cfg["SS_total_max"]:
        errors.append(
            f"📦 **SS^max**: суммарный минимальный страховой запас {ss_min_sum:,.2f} превышает лимит {globals_cfg['SS_total_max']:,.2f}."
        )
    if i_min_sum > capacity:
        errors.append(
            f"📦 **Склад**: минимальный необходимый запас {i_min_sum:,.2f} превышает общую вместимость склада {capacity:,.2f}."
        )
    if c_sum > globals_cfg["C_total_max"]:
        errors.append(
            f"💸 **C^max**: суммарный нормативный показатель затрат {c_sum:,.2f} превышает лимит {globals_cfg['C_total_max']:,.2f}."
        )
    if y_sum < globals_cfg["Y_total_min"]:
        errors.append(
            f"📈 **Y**: суммарная производительность {y_sum:,.2f} ниже требуемого минимума {globals_cfg['Y_total_min']:,.2f}."
        )
    if p_l_sum > globals_cfg["P_total_max"]:
        errors.append(
            f"⚠️ **P^max**: суммарный риск задержек {p_l_sum:,.2f} превышает лимит {globals_cfg['P_total_max']:,.2f}."
        )

    if q_min_sum * 0.0 > budget:
        pass

    min_budget_needed = float((df_input["C"] * df_input["Q_min"]).sum())
    if min_budget_needed > budget:
        errors.append(
            f"💰 **Бюджет**: для закупки обязательного минимума требуется {min_budget_needed:,.2f} руб., "
            f"а доступно только {budget:,.2f} руб."
        )

    for w in warnings:
        st.warning(w)

    return errors

def build_bounds(df_input: pd.DataFrame) -> tuple[list[tuple[float, float]], list[float]]:
    bounds = []
    x0 = []

    for _, row in df_input.iterrows():
        d = float(row["D"])
        q_min = float(row["Q_min"])
        q_max = float(row["Q_max"])
        ss_min = float(row["SS_min"])
        ss_max = float(row["SS_max"])
        i_min = float(row["I_min"])
        w_i = float(row["W_i"])
        b_max = float(row["B_max"])
        n_max = float(row["N_max"])
        m_max = float(row["M_max"])
        t = float(row["T"])
        t_max = float(row["T_max"])

        q_lower = q_min
        if d > 0 and n_max > 0:
            q_lower = max(q_lower, d / n_max)
        if d > 0 and m_max > 0:
            q_lower = max(q_lower, d / m_max)

        q_upper = q_max
        if t > 0 and t_max > 0:
            q_upper = min(q_upper, t_max / t)

        if q_upper < q_lower:
            q_upper = q_lower

        i_lower = i_min
        i_upper = max(i_min, w_i)

        ss_lower = ss_min
        ss_upper = min(ss_max, w_i)
        if ss_upper < ss_lower:
            ss_upper = ss_lower

        b_lower = 0.0
        b_upper = min(b_max, max(d, 0.0))

        bounds.extend([
            (q_lower, q_upper),
            (i_lower, i_upper),
            (ss_lower, ss_upper),
            (b_lower, b_upper),
        ])

        q0 = min(max((q_lower + q_upper) / 2.0, q_lower), q_upper)
        i0 = min(max((i_lower + i_upper) / 2.0, i_lower), i_upper)
        ss0 = min(max((ss_lower + ss_upper) / 2.0, ss_lower), ss_upper)
        b0 = min(max(0.0, b_upper / 2.0), b_upper)

        x0.extend([q0, i0, ss0, b0])

    return bounds, x0

def run_optimization(df_input: pd.DataFrame, mode: str, globals_cfg: dict, weights: dict):
    n = len(df_input)
    bounds, x0 = build_bounds(df_input)

    SCALE = 1_000_000.0

    def objective(x: np.ndarray) -> float:
        val, _, _ = objective_and_metrics(x, df_input, mode, weights)
        return float(val) / SCALE

    cons = []

    cons.append({
        "type": "ineq",
        "fun": lambda x: (globals_cfg["F_budget"] - float(sum(x[i * 4] * float(df_input.iloc[i]["C"]) for i in range(n)))) / SCALE
    })

    cons.append({
        "type": "ineq",
        "fun": lambda x: (globals_cfg["W_total"] - float(sum(x[i * 4 + 1] for i in range(n)))) / 1000.0
    })

    cons.append({
        "type": "ineq",
        "fun": lambda x: (globals_cfg["SS_total_max"] - float(sum(x[i * 4 + 2] for i in range(n)))) / 1000.0
    })

    cons.append({
        "type": "ineq",
        "fun": lambda x: (globals_cfg["Q_total_max"] - float(sum(x[i * 4] for i in range(n)))) / 1000.0
    })

    cons.append({
        "type": "ineq",
        "fun": lambda x: globals_cfg["B_risk_max"] - float(sum(x[i * 4 + 3] * float(df_input.iloc[i]["V"]) for i in range(n)))
    })

    cons.append({
        "type": "ineq",
        "fun": lambda x: (globals_cfg["T_total_max"] - float(sum(float(df_input.iloc[i]["T"]) * x[i * 4] for i in range(n)))) / SCALE
    })

    for i in range(n):
        def bal_con(x, i=i):
            D_val = float(df_input.iloc[i]["D"])
            Q = x[i * 4]
            SS = x[i * 4 + 2]
            B = x[i * 4 + 3]
            return (Q + SS) - B - D_val

        cons.append({"type": "ineq", "fun": bal_con})

    res = minimize(
        objective,
        np.array(x0, dtype=float),
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"maxiter": 3000, "ftol": 1e-9, "disp": False},
    )

    return res

def style_result_table(res_df: pd.DataFrame):
    def color_limits(row):
        q_i = row["Заказ (Q)"]
        i_i = row["Запас (I)"]
        ss_i = row["Страх.запас (SS)"]
        b_i = row["Дефицит (B)"]
        w_i = row["Макс. ВМ (W_i)"]
        budget = row["Бюджет_Лимит"]

        q_style = "background-color: #1e4620; color: white;"
        i_style = "background-color: #1e4620; color: white;"
        ss_style = "background-color: #1e4620; color: white;"
        b_style = "background-color: #1e4620; color: white;"
        cost_style = "background-color: #1e4620; color: white;"

        if q_i + i_i >= w_i * 0.95:
            q_style = "background-color: #8c1d18; color: white; font-weight: bold;"
            i_style = "background-color: #8c1d18; color: white; font-weight: bold;"
        elif q_i + i_i >= w_i * 0.85:
            q_style = "background-color: #b58900; color: white; font-weight: bold;"
            i_style = "background-color: #b58900; color: white; font-weight: bold;"

        if ss_i > 0.8 * w_i:
            ss_style = "background-color: #b58900; color: white; font-weight: bold;"

        if b_i > 0:
            b_style = "background-color: #8c1d18; color: white; font-weight: bold;"

        if row["Затраты (руб)"] > budget:
            cost_style = "background-color: #8c1d18; color: white; font-weight: bold;"
        elif row["Затраты (руб)"] >= budget * 0.8:
            cost_style = "background-color: #b58900; color: white; font-weight: bold;"

        styles = [""] * len(row)
        col_idx = {col: i for i, col in enumerate(row.index)}
        
        styles[col_idx["Заказ (Q)"]] = q_style
        styles[col_idx["Запас (I)"]] = i_style
        styles[col_idx["Страх.запас (SS)"]] = ss_style
        styles[col_idx["Дефицит (B)"]] = b_style
        styles[col_idx["Затраты (руб)"]] = cost_style
        
        return styles

    return (
        res_df.style
        .apply(color_limits, axis=1)
        .format({
            "Заказ (Q)": "{:.2f}",
            "Запас (I)": "{:.2f}",
            "Страх.запас (SS)": "{:.2f}",
            "Дефицит (B)": "{:.2f}",
            "Затраты (руб)": "{:,.2f}",
        })
    )

df_current = load_inventory_df()

with st.sidebar:
    st.header("Настройки оптимизации")
    mode = st.selectbox(
        "Выберите целевую функцию:",
        [
            "F1: Минимизация совокупных затрат",
            "F2: Потери от неопределенности",
            "F3: Макс. уровня обеспечения",
            "F4: Риск логистических сбоев",
            "F5: Транспортно-складские расходы",
            "F6: Макс. эффективности склада",
        ],
    )

    st.divider()
    st.subheader("Глобальные параметры")

    f_budget_str = st.text_input("Общий бюджет закупок (F), руб.", value="5000000")
    try:
        F_budget = float(f_budget_str.replace(" ", "").replace(",", "."))
    except ValueError:
        st.error("Введите корректное число без разделителей тысяч.")
        F_budget = 5_000_000.0

    W_total = st.number_input("Общая емкость склада (W)", min_value=0.0, value=float(df_current["W_i"].sum() * 1.5), step=100.0)

    Q_total_max = st.number_input("Лимит суммарного объёма заказа (Q^max)", min_value=0.0, value=float(df_current["Q_max"].sum()), step=100.0)
    SS_total_max = st.number_input("Лимит суммарного страхового запаса (SS^max)", min_value=0.0, value=float(df_current["SS_max"].sum()), step=100.0)
    B_risk_max = st.number_input("Лимит суммарного риска дефицита ∑V_i·B_i", min_value=0.0, value=float((df_current["V"] * df_current["B_max"]).sum()), step=10.0)
    T_total_max = st.number_input("Лимит транспортных затрат ∑T_i·Q_i", min_value=0.0, value=float((df_current["T"] * df_current["Q_max"]).sum()), step=100.0)
    P_total_max = st.number_input("Лимит логистического риска ∑P_i·L_i", min_value=0.0, value=float((df_current["P"] * df_current["L"]).sum()), step=1.0)
    C_total_max = st.number_input("Лимит суммарных нормативных затрат ∑C_i", min_value=0.0, value=float(df_current["C_max"].sum()), step=100.0)
    Y_total_min = st.number_input("Минимальная суммарная производительность ∑Y_i", min_value=0.0, value=float(df_current["Y_min"].sum()), step=10.0)

    weights = {"alpha": 0.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0, "lambda": 0.0, "theta": 0.0, "R0": 0.95, "C0_factor": 0.9}
    if mode.startswith("F2"):
        st.subheader("Весовые коэффициенты (F2)")
        weights["alpha"] = st.slider("Спрос (alpha)", 0.0, 1.0, 0.30, 0.01)
        weights["beta"] = st.slider("Сроки (beta)", 0.0, 1.0, 0.20, 0.01)
        weights["gamma"] = st.slider("Дефицит (gamma)", 0.0, 1.0, 0.10, 0.01)
        weights["delta"] = st.slider("Страх. запас (delta)", 0.0, 1.0, 0.10, 0.01)
        weights["lambda"] = st.slider("Надежность (lambda)", 0.0, 1.0, 0.20, 0.01)
        weights["theta"] = st.slider("Затраты (theta)", 0.0, 1.0, 0.10, 0.01)

globals_cfg = {
    "F_budget": F_budget,
    "W_total": W_total,
    "Q_total_max": Q_total_max,
    "SS_total_max": SS_total_max,
    "B_risk_max": B_risk_max,
    "T_total_max": T_total_max,
    "P_total_max": P_total_max,
    "C_total_max": C_total_max,
    "Y_total_min": Y_total_min,
}

st.write("### Исходные данные номенклатуры ПАО «КАДВИ»")
st.caption("Поля N_max, M_max, T_max, SS_max и C_max добавлены для полного соответствия постановке.")

column_config = {
    "id": st.column_config.NumberColumn("ID", disabled=True),
    "name": st.column_config.TextColumn("Наименование", required=True),
    "C": st.column_config.NumberColumn("Цена закупки (C)", min_value=0.0),
    "H": st.column_config.NumberColumn("Затраты хранения (H)", min_value=0.0),
    "S": st.column_config.NumberColumn("Затраты оформления заказа (S)", min_value=0.0),
    "T": st.column_config.NumberColumn("Транспортные затраты (T)", min_value=0.0),
    "P": st.column_config.NumberColumn("Вероятность задержки (P)", min_value=0.0, max_value=1.0),
    "R": st.column_config.NumberColumn("Надежность поставщика (R)", min_value=0.0, max_value=1.0),
    "K": st.column_config.NumberColumn("Качество поставок (K)", min_value=0.0, max_value=1.0),
    "D": st.column_config.NumberColumn("Спрос (D)", min_value=0.0),
    "D_fuzzy_min": st.column_config.NumberColumn("D fuzzy min", min_value=0.0),
    "D_fuzzy_max": st.column_config.NumberColumn("D fuzzy max", min_value=0.0),
    "L": st.column_config.NumberColumn("Время поставки (L)", min_value=0.0),
    "L_fuzzy_min": st.column_config.NumberColumn("L fuzzy min", min_value=0.0),
    "L_fuzzy_max": st.column_config.NumberColumn("L fuzzy max", min_value=0.0),
    "W_i": st.column_config.NumberColumn("Вместимость (W_i)", min_value=0.0),
    "N_max": st.column_config.NumberColumn("N_max", min_value=0.0),
    "B_max": st.column_config.NumberColumn("B_max", min_value=0.0),
    "Q_min": st.column_config.NumberColumn("Q_min", min_value=0.0),
    "Q_max": st.column_config.NumberColumn("Q_max", min_value=0.0),
    "SS_min": st.column_config.NumberColumn("SS_min", min_value=0.0),
    "SS_max": st.column_config.NumberColumn("SS_max", min_value=0.0),
    "I_min": st.column_config.NumberColumn("I_min", min_value=0.0),
    "V": st.column_config.NumberColumn("V", min_value=0.0),
    "E": st.column_config.NumberColumn("E", min_value=0.0),
    "Z": st.column_config.NumberColumn("Z", min_value=0.0),
    "M_max": st.column_config.NumberColumn("M_max", min_value=0.0),
    "d_dist": st.column_config.NumberColumn("d_dist", min_value=0.0),
    "T_max": st.column_config.NumberColumn("T_max", min_value=0.0),
    "A": st.column_config.NumberColumn("A", min_value=0.0),
    "G": st.column_config.NumberColumn("G", min_value=0.0),
    "Fi_cost": st.column_config.NumberColumn("Fi_cost", min_value=0.0),
    "U_max": st.column_config.NumberColumn("U_max", min_value=0.0, max_value=1.0),
    "Y_prod": st.column_config.NumberColumn("Y_prod", min_value=0.0),
    "Y_min": st.column_config.NumberColumn("Y_min", min_value=0.0),
    "C_max": st.column_config.NumberColumn("C_max", min_value=0.0),
}

edited_df = st.data_editor(
    df_current,
    num_rows="dynamic",
    hide_index=True,
    column_config=column_config,
    use_container_width=True,
)

db_col1, db_col2, _ = st.columns([1.5, 2, 5])
with db_col1:
    if st.button("💾 Сохранить изменения в БД", type="secondary"):
        try:
            save_inventory_df(edited_df)
            st.success("Данные успешно зафиксированы в SQLite.")
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось сохранить базу: {exc}")

with db_col2:
    if st.button("🔄 Сбросить БД к исходным настройкам", type="secondary"):
        try:
            seed_database(force_reset=True)
            st.rerun()
        except Exception as exc:
            st.error(f"Не удалось сбросить базу: {exc}")

st.divider()

if st.button("🚀 ЗАПУСТИТЬ ОПТИМИЗАЦИОННЫЙ РАСЧЕТ", type="primary"):
    if len(edited_df) == 0:
        st.error("Таблица пуста. Добавьте хотя бы одну позицию номенклатуры.")
        st.stop()

    validation_errors = validate_inputs(edited_df, F_budget, W_total, globals_cfg)
    if validation_errors:
        st.error("⚠️ Обнаружены критические противоречия во введенных данных.")
        for err in validation_errors:
            st.warning(err)
        st.stop()

    with st.spinner("Математический движок выполняет поиск оптимального решения..."):
        result = run_optimization(edited_df, mode, globals_cfg, weights)

    if result is not None and result.success:
        raw_total, metrics, per_item = objective_and_metrics(result.x, edited_df, mode, weights)
        displayed_value = -raw_total if mode.startswith(("F3", "F6")) else raw_total

        st.success(f"🎯 Оптимальный план найден. Значение целевой функции: {displayed_value:,.4f}")

        res_data = []
        for i in range(len(edited_df)):
            idx = i * 4
            row = edited_df.iloc[i]
            Q_val = float(result.x[idx])
            I_val = float(result.x[idx + 1])
            SS_val = float(result.x[idx + 2])
            B_val = float(result.x[idx + 3])

            D = float(row["D"])
            W_i = float(row["W_i"])
            C = float(row["C"])
            T = float(row["T"])
            V = float(row["V"])

            N_i = safe_div(D, max(Q_val, 1e-6))
            M_i = safe_div(D, max(Q_val, 1e-6))
            U_i = min(max(I_val / max(W_i, 1e-6), 0.0), max(float(row["U_max"]), 0.0))

            purchase_cost = C * Q_val
            transport_cost = T * Q_val
            risk_cost = V * B_val

            res_data.append({
                "Товар": normalize_text(row["name"], f"Строка {i + 1}"),
                "Заказ (Q)": Q_val,
                "Запас (I)": I_val,
                "Страх.запас (SS)": SS_val,
                "Дефицит (B)": B_val,
                "N_i": N_i,
                "M_i": M_i,
                "U_i": U_i,
                "Затраты (руб)": purchase_cost,
                "Транспорт (руб)": transport_cost,
                "Риск (руб)": risk_cost,
                "Макс. ВМ (W_i)": W_i,
                "Бюджет_Лимит": F_budget,
            })

        res_df = pd.DataFrame(res_data)

        c1, c2, c3, c4 = st.columns(4)
        total_purchase_cost = float(res_df["Затраты (руб)"].sum())
        total_transport_cost = float(res_df["Транспорт (руб)"].sum())
        total_risk = float(res_df["Риск (руб)"].sum())
        total_stock = float(res_df["Запас (I)"].sum())

        c1.metric("Использовано бюджета", f"{total_purchase_cost:,.0f} руб", f"{(total_purchase_cost / max(F_budget, 1e-9) * 100):.1f}%")
        c2.metric("Загрузка склада", f"{total_stock:,.0f} ед", f"{(total_stock / max(W_total, 1e-9) * 100):.1f}%")
        c3.metric("Транспортные затраты", f"{total_transport_cost:,.0f} руб")
        c4.metric("Риск-дефицит", f"{total_risk:,.0f} руб")

        st.write("#### Оптимальные параметры управления")
        styled_df = style_result_table(res_df)
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_order=["Товар", "Заказ (Q)", "Запас (I)", "Страх.запас (SS)", "Дефицит (B)", "Затраты (руб)", "Транспорт (руб)", "Риск (руб)"],
        )

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            res_df.to_excel(writer, index=False, sheet_name="План_закупок")
        st.download_button(
            "📥 СКАЧАТЬ ПЛАН В EXCEL",
            data=buffer.getvalue(),
            file_name="KADVI_Opt_Plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.divider()

        col_chart1, col_chart2 = st.columns(2)

        with col_chart1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(name="Объем заказа (Q)", x=res_df["Товар"], y=res_df["Заказ (Q)"]))
            fig1.add_trace(go.Bar(name="Запас (I)", x=res_df["Товар"], y=res_df["Запас (I)"]))
            fig1.add_trace(go.Scatter(
                x=res_df["Товар"],
                y=res_df["Макс. ВМ (W_i)"],
                mode="lines+markers",
                name="Предел вместимости (W_i)",
                line=dict(dash="dash", width=2),
                marker=dict(symbol="x", size=8),
            ))
            fig1.update_layout(title="Объем запасов vs Ограничение склада", barmode="group")
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            fig2 = go.Figure(data=[
                go.Bar(
                    name="Затраты закупки",
                    x=["Общие затраты"],
                    y=[total_purchase_cost],
                    text=[f"{total_purchase_cost:,.0f} руб"],
                    textposition="auto",
                )
            ])
            fig2.add_hline(y=F_budget, line_dash="dash", annotation_text="Лимит бюджета", annotation_position="top left")
            fig2.update_layout(
                title="Фактические затраты vs Бюджет",
                yaxis=dict(range=[0, max(F_budget * 1.2, total_purchase_cost * 1.2, 1.0)]),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        st.write("#### Оценка найденного решения по всем критериям (F1 - F6)")
        categories = list(metrics.keys())
        values_raw = [float(metrics[k]) for k in categories]
        max_abs = max(max(abs(v) for v in values_raw), 1e-9)
        values_norm = [abs(v) / max_abs for v in values_raw]

        fig3 = go.Figure(data=go.Scatterpolar(
            r=values_norm,
            theta=categories,
            fill="toself",
            name="Текущее решение",
        ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
            showlegend=False,
            title="Многокритериальный баланс (нормированные значения)",
        )

        col_r1, col_r2 = st.columns([1, 2])
        with col_r1:
            st.dataframe(
                pd.DataFrame({
                    "Критерий": categories,
                    "Значение": [f"{v:,.4f}" for v in values_raw],
                }),
                hide_index=True,
            )
            st.caption("Радар показывает относительный вклад решения в каждый критерий.")
        with col_r2:
            st.plotly_chart(fig3, use_container_width=True)

        st.divider()
        st.write("#### Сводка по позициям")
        st.dataframe(
            pd.DataFrame(per_item)[["Товар", "Q", "I", "SS", "B", "N_i", "M_i", "U_i", "F1_i", "F2_i", "F3_i", "F4_i", "F5_i", "F6_i"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.error("❌ Алгоритму не удалось сойтись к решению.")
        if result is not None:
            st.warning(f"Сообщение solver: {result.message}")
        st.warning(
            """
            💡 Аналитическая сводка:
            1. Жесткие ограничения делают задачу infeasible.
            2. Один из глобальных лимитов слишком мал.
            3. Для какой-то позиции нарушается достижимость Q_min / SS_min / N_max / M_max / T_max.
            """
        )