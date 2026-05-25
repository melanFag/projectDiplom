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
DB_NAME = Path(__file__).with_name("kadvi_model_final.db")

# Строгое соответствие твоему списку
SCHEMA_COLUMNS = [
    "id", "name", 
    "C_i", "H_i", "S_i", "T_i", "P_i", "R_i", "O_i", "W_i", 
    "K_i", "V_i", "E_i", "Z_i", "A_i", "G_i", "F_i", "Q_i", "L_i",
    "I_i", "N_i", "B_i", "SS_i", "M_i", "U_i", "Y_i", "d_i"
]

# Колонки, которые будут строго в таблице справа
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

# 1. SIDEBAR (Слева: Режим, Константы, затем Глобальные переменные)
with st.sidebar:
    st.header("⚙️ Режим расчета")
    mode = st.selectbox("🎯 Целевая функция:", [
        "F1: Минимизация совокупных затрат", "F2: Потери от неопределенности",
        "F3: Макс. уровня обеспечения", "F4: Риск логистических сбоев",
        "F5: Транспортно-складские расходы", "F6: Макс. эффективности склада"
    ])

    st.divider()
    
    # ПОСТОЯННЫЕ ПАРАМЕТРЫ СТРОГО ПЕРЕД ГЛОБАЛЬНЫМИ
    st.markdown("### 🔒 Постоянные параметры")
    selected_idx = st.selectbox(
        "📝 Выбор позиции:", 
        range(len(df_current)), 
        format_func=lambda x: df_current.iloc[x]['name']
    )
    
    # Разбил на 2 колонки, чтобы список не был слишком длинным, но все на виду
    c1, c2 = st.columns(2)
    C_i = c1.number_input("C_i", value=float(df_current.at[selected_idx, "C_i"]))
    H_i = c2.number_input("H_i", value=float(df_current.at[selected_idx, "H_i"]))
    S_i = c1.number_input("S_i", value=float(df_current.at[selected_idx, "S_i"]))
    T_i = c2.number_input("T_i", value=float(df_current.at[selected_idx, "T_i"]))
    P_i = c1.number_input("P_i", value=float(df_current.at[selected_idx, "P_i"]))
    R_i = c2.number_input("R_i", value=float(df_current.at[selected_idx, "R_i"]))
    O_i = c1.number_input("O_i", value=float(df_current.at[selected_idx, "O_i"]))
    W_i = c2.number_input("W_i", value=float(df_current.at[selected_idx, "W_i"]))
    K_i = c1.number_input("K_i", value=float(df_current.at[selected_idx, "K_i"]))
    V_i = c2.number_input("V_i", value=float(df_current.at[selected_idx, "V_i"]))
    E_i = c1.number_input("E_i", value=float(df_current.at[selected_idx, "E_i"]))
    Z_i = c2.number_input("Z_i", value=float(df_current.at[selected_idx, "Z_i"]), help="Риск (второй S_i из ТЗ)")
    A_i = c1.number_input("A_i", value=float(df_current.at[selected_idx, "A_i"]))
    G_i = c2.number_input("G_i", value=float(df_current.at[selected_idx, "G_i"]))
    F_i = c1.number_input("F_i", value=float(df_current.at[selected_idx, "F_i"]))
    Q_i = c2.number_input("Q_i", value=float(df_current.at[selected_idx, "Q_i"]))
    L_i = c1.number_input("L_i", value=float(df_current.at[selected_idx, "L_i"]))

    # Сохранение постоянных значений
    df_current.loc[selected_idx, ["C_i", "H_i", "S_i", "T_i", "P_i", "R_i", "O_i", "W_i", "K_i", "V_i", "E_i", "Z_i", "A_i", "G_i", "F_i", "Q_i", "L_i"]] = [C_i, H_i, S_i, T_i, P_i, R_i, O_i, W_i, K_i, V_i, E_i, Z_i, A_i, G_i, F_i, Q_i, L_i]

    st.divider()

    # ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ПОД КОНСТАНТАМИ
    st.markdown("### 🌍 Глобальные переменные")
    F_budget = st.number_input("F (Бюджет, руб)", value=5000000.0, step=100000.0)
    W_total = st.number_input("W (Склад, ед)", value=10000.0, step=1000.0)


# 2. ГЛАВНЫЙ ЭКРАН: ТАБЛИЦА ПЕРЕМЕННЫХ И КНОПКИ
st.markdown("### 🎛️ Переменные параметры")
# Таблица содержит СТРОГО Q_i, I_i, N_i, B_i, SS_i, L_i, M_i, U_i, Y_i, d_i
edited_table = st.data_editor(
    df_current[TABLE_COLUMNS],
    use_container_width=True,
    hide_index=True,
    num_rows="dynamic",
    column_config={"name": st.column_config.TextColumn("Товар (name)", width="medium")}
)

# Синхронизация изменений таблицы
for col in TABLE_COLUMNS:
    df_current[col] = edited_table[col]

st.write("") 
c_btn1, c_btn2, _ = st.columns([1, 1, 2])
with c_btn1:
    if st.button("💾 Сохранить БД", use_container_width=True):
        save_inventory_df(df_current)
        st.success("Данные успешно сохранены!")
with c_btn2:
    if st.button("🚀 РАССЧИТАТЬ ОПТИМУМ", type="primary", use_container_width=True):
        st.session_state['run_opt'] = True


# --- БЛОК ВЫВОДА РЕЗУЛЬТАТОВ (ГРАФИКИ, МЕТРИКИ, ЭКСЕЛЬ) ---
if st.session_state.get('run_opt', False):
    st.divider()
    st.markdown("### 🏆 Результаты расчета оптимальных переменных")
    
    with st.spinner("Синтез оптимального решения..."):
        res = run_optimization(df_current, mode, F_budget, W_total)
        
    if res.success:
        raw_val, final_metrics = objective_and_metrics(res.x, df_current, mode)
        displayed_value = abs(raw_val)
        
        st.success(f"🎯 Оптимальный план найден. Значение целевой функции: {displayed_value:,.4f}")

        res_data = []
        for i in range(len(df_current)):
            r = df_current.iloc[i]
            idx = i * 4
            Q_opt, I_opt, SS_opt, B_opt = res.x[idx], res.x[idx+1], res.x[idx+2], res.x[idx+3]
            
            # Пересчет зависимых переменных
            N_opt = r["O_i"] / max(Q_opt, 1e-6)
            M_opt = r["O_i"] / max(Q_opt, 1e-6)
            U_opt = I_opt / max(r["W_i"], 1e-6)
            
            res_data.append({
                "Товар": r["name"],
                "Q_i": Q_opt, "I_i": I_opt, "N_i": N_opt, "B_i": B_opt, "SS_i": SS_opt,
                "L_i": r["L_i"], "M_i": M_opt, "U_i": U_opt, "Y_i": r["Y_i"], "d_i": r["d_i"],
                "Затраты (руб)": r["C_i"]*Q_opt + r["H_i"]*I_opt + r["S_i"]*N_opt,
                "Транспорт (руб)": r["T_i"] * Q_opt,
                "Риск (руб)": r["V_i"] * B_opt,
                "W_i": r["W_i"] # Скрыто для графиков
            })
            
        res_df = pd.DataFrame(res_data)
        
        # --- МЕТРИКИ ---
        c1, c2, c3, c4 = st.columns(4)
        total_cost = float(res_df["Затраты (руб)"].sum())
        c1.metric("Использовано бюджета", f"{total_cost:,.0f} руб", f"{(total_cost / max(F_budget, 1e-9) * 100):.1f}%")
        c2.metric("Загрузка склада", f"{float(res_df['I_i'].sum()):,.0f} ед")
        c3.metric("Транспортные затраты", f"{float(res_df['Транспорт (руб)'].sum()):,.0f} руб")
        c4.metric("Риск-дефицит", f"{float(res_df['Риск (руб)'].sum()):,.0f} руб")

        # --- ИДЕАЛЬНАЯ ТАБЛИЦА ---
        st.write("#### Оптимальные параметры управления")
        def highlight_table(s):
            return ['background-color: #1e4620; color: white;' for _ in s]

        st.dataframe(
            res_df.drop(columns=["W_i"]).style.apply(highlight_table, axis=1).format(precision=2), 
            use_container_width=True, 
            hide_index=True
        )

        # --- ЭКСЕЛЬ ---
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            res_df.drop(columns=["W_i"]).to_excel(writer, index=False, sheet_name="План_закупок")
        st.download_button(
            "📥 СКАЧАТЬ ПЛАН В EXCEL",
            data=buffer.getvalue(),
            file_name="KADVI_Opt_Plan.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        # --- ГРАФИКИ PLOTLY ---
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(name="Объем заказа (Q_i)", x=res_df["Товар"], y=res_df["Q_i"]))
            fig1.add_trace(go.Bar(name="Запас (I_i)", x=res_df["Товар"], y=res_df["I_i"]))
            fig1.add_trace(go.Scatter(
                x=res_df["Товар"], y=res_df["W_i"],
                mode="lines+markers", name="Предел вместимости (W_i)",
                line=dict(dash="dash", width=2, color="red")
            ))
            fig1.update_layout(title="Объем запасов vs Ограничение склада", barmode="stack")
            st.plotly_chart(fig1, use_container_width=True)

        with col_chart2:
            fig2 = go.Figure(data=[go.Bar(name="Затраты закупки", x=["Общие затраты"], y=[total_cost], text=[f"{total_cost:,.0f} руб"], textposition="auto")])
            fig2.add_hline(y=F_budget, line_dash="dash", line_color="red", annotation_text="Лимит бюджета", annotation_position="top left")
            fig2.update_layout(title="Фактические затраты vs Бюджет", yaxis=dict(range=[0, max(F_budget * 1.2, total_cost * 1.2, 1.0)]))
            st.plotly_chart(fig2, use_container_width=True)

        # --- АНАЛИТИКА ЦЕЛЕВЫХ ФУНКЦИЙ ---
        st.divider()
        st.write("#### Анализ эффективности по всем целевым критериям (F1 - F6)")
        metrics_df = pd.DataFrame({
            "Критерий": ["F1 (Совокупные затраты)", "F2 (Потери неопределенности)", "F3 (Уровень обеспечения)", "F4 (Риск сбоев)", "F5 (Транспорт и склад)", "F6 (Эффективность склада)"],
            "Значение": [final_metrics["F1"], final_metrics["F2"], final_metrics["F3"], final_metrics["F4"], final_metrics["F5"], final_metrics["F6"]],
            "Направление оптимизации": ["Min", "Min", "Max", "Min", "Min", "Max"]
        })
        st.dataframe(metrics_df.style.format({"Значение": "{:,.4f}"}), use_container_width=True, hide_index=True)

    else:
        st.error("❌ Алгоритму не удалось сойтись к решению. Попробуйте ослабить ограничения бюджета или вместимости.")
