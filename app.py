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

# --- НАСТРОЙКИ БАЗЫ ДАННЫХ И ПЕРЕМЕННЫХ ---
DB_NAME = Path(__file__).with_name("kadvi_model_v2.db")

SCHEMA_COLUMNS = [
    "id", "name", 
    "C_i", "H_i", "S_i", "T_i", "P_i", "R_i", "O_i", "W_i", 
    "K_i", "V_i", "E_i", "Z_i", "A_i", "G_i", "F_i", "Q_i", "L_i",
    "I_i", "N_i", "B_i", "SS_i", "M_i", "U_i", "Y_i", "d_i"
]

TABLE_COLUMNS = ["name", "Q_i", "I_i", "N_i", "B_i", "SS_i", "L_i", "M_i", "U_i", "Y_i", "d_i"]

def seed_rows():
    return pd.DataFrame([
        {
            "name": "Вал коленчатый", "C_i": 2500, "H_i": 120, "S_i": 450, "T_i": 80, 
            "P_i": 0.15, "R_i": 0.95, "O_i": 400, "W_i": 500, "K_i": 0.98, "V_i": 0.5, 
            "E_i": 1.2, "Z_i": 0.8, "A_i": 12, "G_i": 300, "F_i": 500, "Q_i": 200, "L_i": 10,
            "I_i": 50, "N_i": 2, "B_i": 0, "SS_i": 15, "M_i": 2, "U_i": 0.1, "Y_i": 100, "d_i": 150
        },
        {
            "name": "Шестерня", "C_i": 850, "H_i": 40, "S_i": 200, "T_i": 30, 
            "P_i": 0.10, "R_i": 0.98, "O_i": 1200, "W_i": 1000, "K_i": 0.99, "V_i": 0.3, 
            "E_i": 0.9, "Z_i": 0.4, "A_i": 8, "G_i": 150, "F_i": 200, "Q_i": 600, "L_i": 5,
            "I_i": 100, "N_i": 2, "B_i": 0, "SS_i": 40, "M_i": 2, "U_i": 0.1, "Y_i": 200, "d_i": 80
        },
        {
            "name": "Корпус", "C_i": 4200, "H_i": 300, "S_i": 800, "T_i": 250, 
            "P_i": 0.20, "R_i": 0.90, "O_i": 150, "W_i": 300, "K_i": 0.95, "V_i": 0.7, 
            "E_i": 1.5, "Z_i": 1.2, "A_i": 20, "G_i": 600, "F_i": 800, "Q_i": 50, "L_i": 20,
            "I_i": 10, "N_i": 3, "B_i": 0, "SS_i": 10, "M_i": 3, "U_i": 0.03, "Y_i": 50, "d_i": 300
        },
    ])

def ensure_schema(conn):
    conn.execute(f"""
    CREATE TABLE IF NOT EXISTS inventory (
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT,
        {', '.join([f'{c} REAL' for c in SCHEMA_COLUMNS if c not in ['id', 'name']])}
    )
    """)

def load_inventory_df():
    with sqlite3.connect(DB_NAME) as conn:
        ensure_schema(conn)
        df = pd.read_sql_query("SELECT * FROM inventory ORDER BY id", conn)
        if df.empty:
            df = seed_rows()
            df.to_sql("inventory", conn, if_exists="append", index=False)
            df = pd.read_sql_query("SELECT * FROM inventory ORDER BY id", conn)
        return df

def save_inventory_df(df):
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("DELETE FROM inventory")
        df.to_sql("inventory", conn, if_exists="append", index=False)

df_current = load_inventory_df()

# --- ЦЕЛЕВЫЕ ФУНКЦИИ И ОПТИМИЗАТОР ---
def objective_and_metrics(x: np.ndarray, df: pd.DataFrame, mode: str):
    n = len(df)
    metrics = {"F1": 0.0, "F2": 0.0, "F3": 0.0, "F4": 0.0, "F5": 0.0, "F6": 0.0}

    for i in range(n):
        idx = i * 4
        Q, I, SS, B = x[idx], x[idx + 1], x[idx + 2], x[idx + 3]
        r = df.iloc[i]

        Q_safe = max(Q, 1e-6)
        N_calc = r["O_i"] / Q_safe
        M_calc = r["O_i"] / Q_safe
        U_calc = I / max(r["W_i"], 1e-6)

        F1_i = r["C_i"]*Q + r["H_i"]*I + r["S_i"]*N_calc + r["T_i"]*Q + r["P_i"]*B + r["R_i"]*SS
        F2_i = (r["O_i"] - Q)**2 + (r["W_i"] - I)**2 + B**2
        F3_i = ((Q + SS - B) * r["R_i"] * r["K_i"]) / max(r["O_i"] + r["L_i"], 1e-6)
        F4_i = r["P_i"] * r["L_i"] * Q + r["V_i"] * B + r["E_i"] * M_calc + (r["Z_i"] / max(r["R_i"], 1e-6))
        F5_i = r["A_i"] * r["d_i"] + r["G_i"] * M_calc + r["F_i"] * U_calc
        F6_i = (I * U_calc * r["Y_i"]) / max(r["W_i"] + r["C_i"] + r["L_i"], 1e-6)

        metrics["F1"] += F1_i; metrics["F2"] += F2_i; metrics["F3"] += F3_i
        metrics["F4"] += F4_i; metrics["F5"] += F5_i; metrics["F6"] += F6_i

    if mode.startswith("F1"): return metrics["F1"], metrics
    elif mode.startswith("F2"): return metrics["F2"], metrics
    elif mode.startswith("F3"): return -metrics["F3"], metrics
    elif mode.startswith("F4"): return metrics["F4"], metrics
    elif mode.startswith("F5"): return metrics["F5"], metrics
    elif mode.startswith("F6"): return -metrics["F6"], metrics
    return metrics["F1"], metrics

def run_optimization(df_input, mode, F_budget, W_total):
    n = len(df_input)
    bounds, x0 = [], []
    
    for _, r in df_input.iterrows():
        bounds.extend([(1, r["O_i"]*2), (0, r["W_i"]), (0, r["W_i"]), (0, r["O_i"])])
        x0.extend([max(r["Q_i"], 1), r["I_i"], r["SS_i"], r["B_i"]])

    def objective(x):
        val, _ = objective_and_metrics(x, df_input, mode)
        return val / 1_000_000.0

    cons = [
        {"type": "ineq", "fun": lambda x: F_budget - sum(x[i*4] * df_input.iloc[i]["C_i"] for i in range(n))},
        {"type": "ineq", "fun": lambda x: W_total - sum(x[i*4+1] for i in range(n))}
    ]

    return minimize(objective, np.array(x0), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 1000})


# --- ИНТЕРФЕЙС ПОЛЬЗОВАТЕЛЯ ---

# 1. SIDEBAR (Настройки, Глобальные лимиты и Постоянные параметры)
with st.sidebar:
    st.header("⚙️ Режим расчета")
    mode = st.selectbox("🎯 Целевая функция:", [
        "F1: Минимизация совокупных затрат", "F2: Потери от неопределенности",
        "F3: Макс. уровня обеспечения", "F4: Риск логистических сбоев",
        "F5: Транспортно-складские расходы", "F6: Макс. эффективности склада"
    ])

    st.divider()
    st.markdown("### 🌍 Глобальные лимиты")
    F_budget = st.number_input("F (Бюджет, руб)", value=5000000.0, step=100000.0, help="Общий бюджет закупок")
    W_total = st.number_input("W (Склад, ед)", value=10000.0, step=1000.0, help="Общая емкость склада")

    st.divider()
    st.markdown("### 🔒 Постоянные параметры")
    selected_idx = st.selectbox(
        "📝 Выбор позиции:", 
        range(len(df_current)), 
        format_func=lambda x: df_current.iloc[x]['name']
    )
    
    # Выпадающий список (expander) под глобальными лимитами
    with st.expander("⚙️ Значения констант (Глобальные)"):
        # Две колонки для компактности в сайдбаре
        c1, c2 = st.columns(2)
        
        C_i = c1.number_input("C_i", value=float(df_current.at[selected_idx, "C_i"]), help="Цена закупки")
        H_i = c2.number_input("H_i", value=float(df_current.at[selected_idx, "H_i"]), help="Затраты на хранение")
        
        S_i = c1.number_input("S_i", value=float(df_current.at[selected_idx, "S_i"]), help="Стоимость оформления")
        T_i = c2.number_input("T_i", value=float(df_current.at[selected_idx, "T_i"]), help="Транспортные расходы")
        
        P_i = c1.number_input("P_i", value=float(df_current.at[selected_idx, "P_i"]), help="Вероятность задержки (0-1)")
        R_i = c2.number_input("R_i", value=float(df_current.at[selected_idx, "R_i"]), help="Надежность поставщика (0-1)")
        
        O_i = c1.number_input("O_i", value=float(df_current.at[selected_idx, "O_i"]), help="Спрос")
        W_i = c2.number_input("W_i", value=float(df_current.at[selected_idx, "W_i"]), help="Вместимость (лимит)")
        
        K_i = c1.number_input("K_i", value=float(df_current.at[selected_idx, "K_i"]), help="Качество поставки (0-1)")
        V_i = c2.number_input("V_i", value=float(df_current.at[selected_idx, "V_i"]), help="Штраф за дефицит")
        
        E_i = c1.number_input("E_i", value=float(df_current.at[selected_idx, "E_i"]), help="Эксплуатационные расходы")
        Z_i = c2.number_input("Z_i", value=float(df_current.at[selected_idx, "Z_i"]), help="Риск поставщика")
        
        A_i = c1.number_input("A_i", value=float(df_current.at[selected_idx, "A_i"]), help="Стоимость за 1 км")
        G_i = c2.number_input("G_i", value=float(df_current.at[selected_idx, "G_i"]), help="Стоимость погрузки/разгрузки")
        
        F_i = c1.number_input("F_i", value=float(df_current.at[selected_idx, "F_i"]), help="Стоимость оборудования")
        Q_i = c2.number_input("Q_i", value=float(df_current.at[selected_idx, "Q_i"]), help="Исходный заказ")
        
        L_i = c1.number_input("L_i", value=float(df_current.at[selected_idx, "L_i"]), help="Время поставки (дни)")

        # Сохранение значений
        df_current.loc[selected_idx, ["C_i", "H_i", "S_i", "T_i", "P_i", "R_i", "O_i", "W_i", "K_i", "V_i", "E_i", "Z_i", "A_i", "G_i", "F_i", "Q_i", "L_i"]] = [C_i, H_i, S_i, T_i, P_i, R_i, O_i, W_i, K_i, V_i, E_i, Z_i, A_i, G_i, F_i, Q_i, L_i]


# 2. ГЛАВНЫЙ ЭКРАН: ПЕРЕМЕННЫЕ (ТАБЛИЦА И КНОПКИ)
st.markdown("#### 🎛️ Переменные параметры (Ограничения)")
edited_table = st.data_editor(
    df_current[TABLE_COLUMNS],
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    column_config={"name": st.column_config.TextColumn("Товар (name)", width="medium")}
)

# Синхронизация изменений таблицы в общую базу
for col in TABLE_COLUMNS:
    df_current[col] = edited_table[col]

st.write("") # Небольшой отступ
c_btn1, c_btn2, _ = st.columns([1, 1, 2])
with c_btn1:
    if st.button("💾 Сохранить БД", use_container_width=True):
        save_inventory_df(df_current)
        st.success("Данные успешно сохранены!")
with c_btn2:
    if st.button("🚀 РАССЧИТАТЬ ОПТИМУМ", type="primary", use_container_width=True):
        st.session_state['run_opt'] = True

# --- БЛОК ВЫВОДА РЕЗУЛЬТАТОВ ---
if st.session_state.get('run_opt', False):
    st.divider()
    st.markdown("### 🏆 Результаты расчета оптимальных переменных")
    
    with st.spinner("Поиск идеальных переменных..."):
        res = run_optimization(df_current, mode, F_budget, W_total)
        
    if res.success:
        raw_val, final_metrics = objective_and_metrics(res.x, df_current, mode)
        
        res_data = []
        for i in range(len(df_current)):
            r = df_current.iloc[i]
            idx = i * 4
            Q_opt, I_opt, SS_opt, B_opt = res.x[idx], res.x[idx+1], res.x[idx+2], res.x[idx+3]
            
            # Пересчет зависимых переменных для вывода
            N_opt = r["O_i"] / max(Q_opt, 1e-6)
            M_opt = r["O_i"] / max(Q_opt, 1e-6)
            U_opt = I_opt / max(r["W_i"], 1e-6)
            
            res_data.append({
                "Товар": r["name"],
                "Q_i": Q_opt, "I_i": I_opt, "N_i": N_opt, "B_i": B_opt, "SS_i": SS_opt,
                "L_i": r["L_i"], "M_i": M_opt, "U_i": U_opt, "Y_i": r["Y_i"], "d_i": r["d_i"],
                "Затраты (F1)": r["C_i"]*Q_opt + r["H_i"]*I_opt + r["S_i"]*N_opt
            })
            
        res_df = pd.DataFrame(res_data)
        
        st.dataframe(
            res_df.style.format(precision=2), 
            use_container_width=True, 
            hide_index=True,
            column_order=["Товар", "Q_i", "I_i", "N_i", "B_i", "SS_i", "L_i", "M_i", "U_i", "Y_i", "d_i", "Затраты (F1)"]
        )
        
        st.success(f"Оптимизация успешна! Значение выбранной функции: {abs(raw_val):,.2f}")
    else:
        st.error("Алгоритм не сошелся. Попробуйте увеличить глобальные рамки (Бюджет F или Емкость W).")
