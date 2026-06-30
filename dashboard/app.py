import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================
# Dashboard JRC - Telemetría Volvo
# Grupo C: Visualización, BI y Documentación
# ============================================================

st.set_page_config(
    page_title="JRC | Dashboard Ejecutivo de Flota Volvo",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Estilos generales
# -----------------------------
st.markdown(
    """
    <style>
        .main .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .section-note {
            color: #4B5563;
            font-size: 0.95rem;
            margin-top: -0.3rem;
            margin-bottom: 1.1rem;
        }
        .small-caption {
            color: #6B7280;
            font-size: 0.82rem;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Utilidades
# -----------------------------
def first_existing_path(candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return None


def to_numeric_safe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_datetime_safe(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    return df


def bool_safe(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "yes", "si", "sí"])


@st.cache_data(show_spinner=True)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    return df


def preprocess_analitico(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df = to_datetime_safe(df, ["fecha_creacion_evento", "fecha_recepcion_servidor", "fecha_hora_gps"])

    if "tiene_datos_acumulados" in df.columns:
        df["tiene_datos_acumulados"] = bool_safe(df["tiene_datos_acumulados"])

    numeric_cols = [
        "delta_odometro_km", "delta_combustible_gal", "kpi_rendimiento_gal_km",
        "kpi_utilizacion_efectiva_pct", "flag_alerta_critica", "flag_alerta_preventiva",
        "flag_outlier_consumo", "combustible_total_gal", "combustible_en_ralenti_gal",
        "combustible_en_movimiento_gal", "combustible_con_control_crucero_gal",
        "odometro_total_km", "distancia_recorrida_frenando_km",
        "distancia_para_proximo_servicio_km", "altitud_m",
        "distancia_con_control_crucero_km", "tiempo_motor_encendido_detenido_h",
        "tiempo_en_movimiento_h", "tiempo_con_control_crucero_h", "velocidad_gps",
        "velocidad_ruedas_tacografo", "nivel_combustible_porcentaje", "nivel_adblue_porcentaje",
        "rpm_instantaneas_motor", "temperatura_refrigerante_motor", "horas_totales_motor",
        "peso_bruto_combinado", "hora",
    ]
    df = to_numeric_safe(df, numeric_cols)

    for flag in ["flag_alerta_critica", "flag_alerta_preventiva", "flag_outlier_consumo"]:
        if flag in df.columns:
            df[flag] = df[flag].fillna(0).astype(int)

    return df


def preprocess_acumulado(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "fecha" in df.columns:
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce").dt.date
    df = to_datetime_safe(df, ["fecha_creacion_evento", "fecha_recepcion_servidor", "fecha_hora_gps"])

    if "tiene_datos_acumulados" in df.columns:
        df["tiene_datos_acumulados"] = bool_safe(df["tiene_datos_acumulados"])

    numeric_cols = [
        "tiempo_zona_verde_h", "tiempo_fuera_zona_verde_h", "tiempo_sobre_revolucion_h",
        "tiempo_sobrecarga_motor_h", "tiempo_exceso_velocidad_h",
        "delta_tiempo_zona_verde_h", "delta_tiempo_fuera_zona_verde_h",
        "delta_tiempo_sobre_revolucion_h", "delta_tiempo_sobrecarga_motor_h",
        "delta_tiempo_exceso_velocidad_h", "conteo_frenadas_en_movimiento",
        "conteo_total_frenadas_volvo", "conteo_freno_sin_retardador",
        "distancia_recorrida_freno_sin_retardador", "tiempo_freno_sin_retardador",
        "tiempo_en_movimiento_h", "tiempo_motor_encendido_detenido_h",
        "tiempo_con_control_crucero_h", "distancia_con_control_crucero_km",
        "odometro_total_km", "combustible_total_gal", "velocidad_gps",
        "horas_totales_motor", "hora",
    ]
    df = to_numeric_safe(df, numeric_cols)
    return df


def filter_data(
    df: pd.DataFrame,
    vehicles: list,
    date_range: tuple,
    triggers: list,
    only_accumulated: bool,
) -> pd.DataFrame:
    out = df.copy()

    if vehicles and "id_vehiculo" in out.columns:
        out = out[out["id_vehiculo"].isin(vehicles)]

    if "fecha" in out.columns and date_range and len(date_range) == 2:
        start, end = date_range
        out = out[(out["fecha"] >= start) & (out["fecha"] <= end)]

    if triggers and "tipo_evento_disparador" in out.columns:
        out = out[out["tipo_evento_disparador"].isin(triggers)]

    if only_accumulated and "tiene_datos_acumulados" in out.columns:
        out = out[out["tiene_datos_acumulados"] == True]

    return out


def safe_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def safe_mean(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean())


def format_number(value, decimals=1, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    try:
        value = float(value)
    except Exception:
        return "—"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.{decimals}f}M{suffix}"
    if abs(value) >= 1_000:
        return f"{value/1_000:.{decimals}f}K{suffix}"
    return f"{value:.{decimals}f}{suffix}"


def get_latest_by_vehicle(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "id_vehiculo" not in df.columns:
        return pd.DataFrame()
    sort_col = "fecha_creacion_evento" if "fecha_creacion_evento" in df.columns else "fecha"
    temp = df.sort_values(["id_vehiculo", sort_col])
    return temp.groupby("id_vehiculo", as_index=False).tail(1)


def plot_empty(message="No hay datos suficientes para este gráfico."):
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, showarrow=False, font=dict(size=16))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def bar_top(df: pd.DataFrame, x: str, y: str, title: str, top_n: int = 10):
    if df.empty or x not in df.columns or y not in df.columns:
        return plot_empty()
    temp = df[[x, y]].dropna().sort_values(y, ascending=False).head(top_n)
    if temp.empty:
        return plot_empty()
    fig = px.bar(temp.sort_values(y), x=y, y=x, orientation="h", title=title)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), yaxis_title="", xaxis_title="")
    return fig


# -----------------------------
# Funciones nuevas para KPIs ejecutivos
# -----------------------------
def assign_guardia(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna 'guardia' según la hora del evento (Día: 07-18, Noche: 19-06)."""
    if "hora" not in df.columns:
        df = df.copy()
        df["guardia"] = "Sin datos"
        return df
    df = df.copy()
    hora = pd.to_numeric(df["hora"], errors="coerce")
    df["guardia"] = np.where((hora >= 7) & (hora < 19), "Día", "Noche")
    return df


def _delta_col(grp, col: str) -> pd.Series:
    """Delta (max - min) de una columna acumulada por grupo."""
    return (grp[col].max() - grp[col].min()).clip(lower=0)


def build_vehicle_kpis(df_ana: pd.DataFrame) -> pd.DataFrame:
    """Tabla de KPIs ejecutivos por vehículo."""
    if df_ana.empty or "id_vehiculo" not in df_ana.columns:
        return pd.DataFrame()

    grp = df_ana.groupby("id_vehiculo")
    result = pd.DataFrame({"id_vehiculo": sorted(df_ana["id_vehiculo"].dropna().unique())})

    # Velocidad promedio (solo registros con movimiento)
    if "velocidad_gps" in df_ana.columns:
        vel = (
            df_ana[df_ana["velocidad_gps"] > 0]
            .groupby("id_vehiculo")["velocidad_gps"]
            .mean()
            .reset_index()
            .rename(columns={"velocidad_gps": "velocidad_promedio_kmh"})
        )
        result = result.merge(vel, on="id_vehiculo", how="left")
    else:
        result["velocidad_promedio_kmh"] = np.nan

    # Ralentí (delta de tiempo_motor_encendido_detenido_h por vehículo)
    if "tiempo_motor_encendido_detenido_h" in df_ana.columns:
        idle = _delta_col(grp, "tiempo_motor_encendido_detenido_h").reset_index()
        idle.columns = ["id_vehiculo", "ralenti_h"]
        result = result.merge(idle, on="id_vehiculo", how="left")
    else:
        result["ralenti_h"] = np.nan

    # Movimiento (delta de tiempo_en_movimiento_h por vehículo)
    if "tiempo_en_movimiento_h" in df_ana.columns:
        mov = _delta_col(grp, "tiempo_en_movimiento_h").reset_index()
        mov.columns = ["id_vehiculo", "movimiento_h"]
        result = result.merge(mov, on="id_vehiculo", how="left")
    else:
        result["movimiento_h"] = np.nan

    # Horas totales de motor en el periodo (delta de horas_totales_motor)
    if "horas_totales_motor" in df_ana.columns:
        motor = _delta_col(grp, "horas_totales_motor").reset_index()
        motor.columns = ["id_vehiculo", "horas_motor_h"]
        result = result.merge(motor, on="id_vehiculo", how="left")
    else:
        result["horas_motor_h"] = np.nan

    # Consumo L/hr = (sum delta_combustible_gal * 3.78541) / delta horas_totales_motor
    if "delta_combustible_gal" in df_ana.columns:
        fuel = grp["delta_combustible_gal"].sum().reset_index()
        fuel.columns = ["id_vehiculo", "fuel_gal"]
        result = result.merge(fuel, on="id_vehiculo", how="left")
    else:
        result["fuel_gal"] = np.nan

    if "horas_motor_h" in result.columns and "fuel_gal" in result.columns:
        result["consumo_L_hr"] = np.where(
            result["horas_motor_h"] > 0,
            result["fuel_gal"] * 3.78541 / result["horas_motor_h"],
            np.nan,
        )
    else:
        result["consumo_L_hr"] = np.nan

    return result


def build_guardia_kpis(df_ana: pd.DataFrame) -> pd.DataFrame:
    """KPIs promediados por guardia (Día / Noche), usando delta de acumulados por (vehículo, guardia)."""
    if df_ana.empty or "guardia" not in df_ana.columns or "id_vehiculo" not in df_ana.columns:
        return pd.DataFrame()

    grp = df_ana.groupby(["id_vehiculo", "guardia"])
    rows = []

    for (vid, guardia), sub in grp:
        row = {"id_vehiculo": vid, "guardia": guardia}

        # Velocidad promedio
        if "velocidad_gps" in sub.columns:
            moving = sub[sub["velocidad_gps"] > 0]["velocidad_gps"]
            row["velocidad_kmh"] = moving.mean() if not moving.empty else np.nan
        else:
            row["velocidad_kmh"] = np.nan

        # Ralentí horas
        if "tiempo_motor_encendido_detenido_h" in sub.columns:
            row["ralenti_h"] = max(sub["tiempo_motor_encendido_detenido_h"].max() - sub["tiempo_motor_encendido_detenido_h"].min(), 0)
        else:
            row["ralenti_h"] = np.nan

        # Horas motor
        if "horas_totales_motor" in sub.columns:
            row["horas_motor_h"] = max(sub["horas_totales_motor"].max() - sub["horas_totales_motor"].min(), 0)
        else:
            row["horas_motor_h"] = np.nan

        # Consumo L/hr
        fuel_gal = sub["delta_combustible_gal"].sum() if "delta_combustible_gal" in sub.columns else np.nan
        hm = row.get("horas_motor_h", 0) or 0
        row["consumo_L_hr"] = (fuel_gal * 3.78541 / hm) if (hm > 0 and not pd.isna(fuel_gal)) else np.nan

        rows.append(row)

    detail = pd.DataFrame(rows)
    if detail.empty:
        return pd.DataFrame()

    # Promedio por guardia (una fila por guardia)
    summary = detail.groupby("guardia", as_index=False).agg(
        velocidad_kmh=("velocidad_kmh", "mean"),
        ralenti_h=("ralenti_h", "mean"),
        horas_motor_h=("horas_motor_h", "mean"),
        consumo_L_hr=("consumo_L_hr", "mean"),
        vehiculos=("id_vehiculo", "count"),
    )
    return summary


# -----------------------------
# Carga de datos
# -----------------------------
st.title("Dashboard Ejecutivo de Flota — JRC Volvo")
st.markdown(
    "<div class='section-note'>Indicadores clave de rendimiento operativo de la flota de volquetes.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Configuración")
    st.caption("Coloca los CSV dentro de una carpeta `data/` o ajusta las rutas manualmente.")

    default_ana = first_existing_path([
        "data/dataset_master_kpis_analitico.csv",
        "data/sample_analitico.csv",
        "dataset_master_kpis_analitico.csv",
        "sample_analitico.csv",
        "/mnt/data/sample_analitico.csv",
    ]) or "data/dataset_master_kpis_analitico.csv"

    default_acu = first_existing_path([
        "data/dataset_master_kpis_acumulado.csv",
        "data/sample_acumulado.csv",
        "dataset_master_kpis_acumulado.csv",
        "sample_acumulado.csv",
        "/mnt/data/sample_acumulado.csv",
    ]) or "data/dataset_master_kpis_acumulado.csv"

    path_analitico = st.text_input("Ruta dataset analítico", value=default_ana)
    path_acumulado = st.text_input("Ruta dataset acumulado/motor", value=default_acu)

    load_button = st.button("Cargar / recargar datos", use_container_width=True)

if not Path(path_analitico).exists():
    st.error(f"No se encontró el archivo analítico: {path_analitico}")
    st.stop()

if not Path(path_acumulado).exists():
    st.warning(f"No se encontró el archivo acumulado/motor: {path_acumulado}. Algunas métricas podrían no estar disponibles.")

try:
    df_analitico = preprocess_analitico(load_csv(path_analitico))
except Exception as exc:
    st.error(f"Error leyendo dataset analítico: {exc}")
    st.stop()

try:
    if Path(path_acumulado).exists():
        df_acumulado = preprocess_acumulado(load_csv(path_acumulado))
    else:
        df_acumulado = pd.DataFrame()
except Exception as exc:
    st.error(f"Error leyendo dataset acumulado/motor: {exc}")
    st.stop()


# -----------------------------
# Filtros globales
# -----------------------------
with st.sidebar:
    st.header("Filtros")

    all_vehicles = sorted(df_analitico["id_vehiculo"].dropna().unique().tolist()) if "id_vehiculo" in df_analitico.columns else []
    selected_vehicles = st.multiselect(
        "Vehículos",
        options=all_vehicles,
        default=all_vehicles,
        help="Selecciona uno o varios vehículos para comparar.",
    )

    if "fecha" in df_analitico.columns and df_analitico["fecha"].notna().any():
        min_date = min(df_analitico["fecha"].dropna())
        max_date = max(df_analitico["fecha"].dropna())
        selected_dates = st.date_input("Rango de fechas", value=(min_date, max_date), min_value=min_date, max_value=max_date)
        if not isinstance(selected_dates, tuple) or len(selected_dates) != 2:
            selected_dates = (min_date, max_date)
    else:
        selected_dates = None

    all_triggers = sorted(df_analitico["tipo_evento_disparador"].dropna().unique().tolist()) if "tipo_evento_disparador" in df_analitico.columns else []
    selected_triggers = st.multiselect(
        "Tipo de evento",
        options=all_triggers,
        default=all_triggers,
    )

    only_accumulated = st.checkbox(
        "Usar solo registros con datos acumulados",
        value=False,
        help="Actívalo para KPIs que dependen de accumulatedData.",
    )

    st.divider()
    st.caption("Los KPIs de consumo y ralentí se calculan como delta de valores acumulados por vehículo.")

f_ana = filter_data(df_analitico, selected_vehicles, selected_dates, selected_triggers, only_accumulated)
f_acu = filter_data(df_acumulado, selected_vehicles, selected_dates, selected_triggers if not df_acumulado.empty else [], False)

if f_ana.empty:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# Asignar guardia
f_ana = assign_guardia(f_ana)

# Construir tablas de KPIs
vehicle_kpis = build_vehicle_kpis(f_ana)
guardia_kpis = build_guardia_kpis(f_ana)


# -----------------------------
# KPIs ejecutivos (tarjetas)
# -----------------------------
vehiculos = f_ana["id_vehiculo"].nunique() if "id_vehiculo" in f_ana.columns else 0

# Velocidad promedio de flota
if "velocidad_gps" in f_ana.columns:
    vel_data = f_ana[f_ana["velocidad_gps"] > 0]["velocidad_gps"]
    velocidad_flota = vel_data.mean() if not vel_data.empty else np.nan
else:
    velocidad_flota = np.nan

# Ralentí promedio por vehículo
ralenti_promedio = vehicle_kpis["ralenti_h"].mean() if not vehicle_kpis.empty and "ralenti_h" in vehicle_kpis.columns else np.nan

# Consumo L/hr de flota
total_fuel_L = safe_sum(f_ana, "delta_combustible_gal") * 3.78541
total_motor_h = vehicle_kpis["horas_motor_h"].sum() if not vehicle_kpis.empty and "horas_motor_h" in vehicle_kpis.columns else 0
consumo_L_hr_flota = total_fuel_L / total_motor_h if total_motor_h > 0 else np.nan

# Horas totales de motor
horas_motor_total = vehicle_kpis["horas_motor_h"].sum() if not vehicle_kpis.empty and "horas_motor_h" in vehicle_kpis.columns else np.nan

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Volquetes", f"{vehiculos}")
c2.metric("Velocidad promedio", format_number(velocidad_flota, 1, " km/h"))
c3.metric("Ralentí promedio", format_number(ralenti_promedio, 1, " h"))
c4.metric("Consumo L/hr", format_number(consumo_L_hr_flota, 2, " L/hr"))
c5.metric("Horas de motor", format_number(horas_motor_total, 1, " h"))

st.divider()


# -----------------------------
# Pestañas
# -----------------------------
tab_resumen, tab_rendimiento, tab_guardias, tab_conclusiones = st.tabs([
    "1. Resumen Ejecutivo",
    "2. Rendimiento por Volquete",
    "3. Comparación de Guardias",
    "4. Conclusiones",
])


# ---- 1. Resumen Ejecutivo ----
with tab_resumen:
    st.subheader("Resumen Ejecutivo")
    st.markdown(
        "<div class='section-note'>Evolución diaria de la utilización de la flota.</div>",
        unsafe_allow_html=True,
    )

    # Evolución diaria de horas totales de motor
    if "fecha" in f_ana.columns and "horas_totales_motor" in f_ana.columns:
        # Delta de horas de motor por (vehículo, fecha) → suma diaria de la flota
        daily_motor = (
            f_ana.groupby(["id_vehiculo", "fecha"])["horas_totales_motor"]
            .agg(["max", "min"])
            .reset_index()
        )
        daily_motor["delta_h"] = (daily_motor["max"] - daily_motor["min"]).clip(lower=0)
        daily_sum = daily_motor.groupby("fecha", as_index=False)["delta_h"].sum()
        daily_sum = daily_sum.sort_values("fecha")

        fig_motor_diario = px.line(
            daily_sum,
            x="fecha",
            y="delta_h",
            markers=True,
            title="Evolución diaria — Horas totales de motor (flota)",
            labels={"fecha": "Fecha", "delta_h": "Horas de motor"},
        )
        fig_motor_diario.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=60, b=10),
            xaxis_title="Fecha",
            yaxis_title="Horas de motor",
        )
        st.plotly_chart(fig_motor_diario, use_container_width=True)
    else:
        st.plotly_chart(plot_empty("No hay datos de horas de motor disponibles."), use_container_width=True)

    # Tabla resumen por vehículo
    st.subheader("Resumen de KPIs por volquete")
    if not vehicle_kpis.empty:
        display_cols = {
            "id_vehiculo": "Vehículo",
            "velocidad_promedio_kmh": "Velocidad prom. (km/h)",
            "ralenti_h": "Ralentí (h)",
            "movimiento_h": "Movimiento (h)",
            "horas_motor_h": "Horas motor (h)",
            "consumo_L_hr": "Consumo (L/hr)",
        }
        cols_available = [c for c in display_cols if c in vehicle_kpis.columns]
        display_df = vehicle_kpis[cols_available].rename(columns=display_cols).round(2)
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No hay datos suficientes para construir el resumen.")


# ---- 2. Rendimiento por Volquete ----
with tab_rendimiento:
    st.subheader("Rendimiento por Volquete")
    st.markdown(
        "<div class='section-note'>Comparación de velocidad, consumo y ralentí por vehículo.</div>",
        unsafe_allow_html=True,
    )

    # --- Velocidad promedio ---
    st.markdown("#### Velocidad promedio (km/h)")
    if not vehicle_kpis.empty and "velocidad_promedio_kmh" in vehicle_kpis.columns:
        temp_vel = vehicle_kpis[["id_vehiculo", "velocidad_promedio_kmh"]].dropna().sort_values("velocidad_promedio_kmh", ascending=False)
        if not temp_vel.empty:
            fig_vel = px.bar(
                temp_vel.sort_values("velocidad_promedio_kmh"),
                x="velocidad_promedio_kmh",
                y="id_vehiculo",
                orientation="h",
                title="Velocidad promedio por volquete",
                labels={"velocidad_promedio_kmh": "km/h", "id_vehiculo": "Vehículo"},
            )
            fig_vel.update_layout(height=max(350, len(temp_vel) * 38), margin=dict(l=10, r=10, t=60, b=10), yaxis_title="", xaxis_title="km/h")
            st.plotly_chart(fig_vel, use_container_width=True)
        else:
            st.plotly_chart(plot_empty("Sin datos de velocidad."), use_container_width=True)
    else:
        st.plotly_chart(plot_empty("Sin datos de velocidad."), use_container_width=True)

    st.divider()

    # --- Consumo L/hr ---
    st.markdown("#### Consumo de combustible (L/hr)")
    col_a, col_b = st.columns(2)

    with col_a:
        if not vehicle_kpis.empty and "consumo_L_hr" in vehicle_kpis.columns:
            temp_cons = vehicle_kpis[["id_vehiculo", "consumo_L_hr"]].dropna().sort_values("consumo_L_hr", ascending=False)
            if not temp_cons.empty:
                fig_cons = px.bar(
                    temp_cons.sort_values("consumo_L_hr"),
                    x="consumo_L_hr",
                    y="id_vehiculo",
                    orientation="h",
                    title="Consumo por volquete (L/hr)",
                    labels={"consumo_L_hr": "L/hr", "id_vehiculo": "Vehículo"},
                )
                fig_cons.update_layout(height=max(350, len(temp_cons) * 38), margin=dict(l=10, r=10, t=60, b=10), yaxis_title="", xaxis_title="L/hr")
                st.plotly_chart(fig_cons, use_container_width=True)
            else:
                st.plotly_chart(plot_empty("Sin datos de consumo L/hr."), use_container_width=True)
        else:
            st.plotly_chart(plot_empty("Sin datos de consumo L/hr."), use_container_width=True)

    with col_b:
        # Evolución diaria del consumo L/hr de la flota
        if "fecha" in f_ana.columns and "delta_combustible_gal" in f_ana.columns and "horas_totales_motor" in f_ana.columns:
            daily_fuel = f_ana.groupby(["id_vehiculo", "fecha"]).agg(
                fuel_gal=("delta_combustible_gal", "sum"),
                motor_max=("horas_totales_motor", "max"),
                motor_min=("horas_totales_motor", "min"),
            ).reset_index()
            daily_fuel["delta_h"] = (daily_fuel["motor_max"] - daily_fuel["motor_min"]).clip(lower=0)
            daily_total = daily_fuel.groupby("fecha", as_index=False).agg(
                fuel_gal=("fuel_gal", "sum"),
                delta_h=("delta_h", "sum"),
            )
            daily_total["consumo_L_hr"] = np.where(
                daily_total["delta_h"] > 0,
                daily_total["fuel_gal"] * 3.78541 / daily_total["delta_h"],
                np.nan,
            )
            daily_total = daily_total.sort_values("fecha")
            fig_cons_diario = px.line(
                daily_total,
                x="fecha",
                y="consumo_L_hr",
                markers=True,
                title="Evolución diaria — Consumo medio (L/hr)",
                labels={"fecha": "Fecha", "consumo_L_hr": "L/hr"},
            )
            fig_cons_diario.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10), xaxis_title="Fecha", yaxis_title="L/hr")
            st.plotly_chart(fig_cons_diario, use_container_width=True)
        else:
            st.plotly_chart(plot_empty("Sin datos para evolución de consumo."), use_container_width=True)

    st.divider()

    # --- Tiempo en ralentí ---
    st.markdown("#### Tiempo en ralentí")
    col_c, col_d = st.columns(2)

    with col_c:
        if not vehicle_kpis.empty and "ralenti_h" in vehicle_kpis.columns:
            temp_idle = vehicle_kpis[["id_vehiculo", "ralenti_h"]].dropna().sort_values("ralenti_h", ascending=False)
            if not temp_idle.empty:
                fig_idle = px.bar(
                    temp_idle.sort_values("ralenti_h"),
                    x="ralenti_h",
                    y="id_vehiculo",
                    orientation="h",
                    title="Ranking — Tiempo en ralentí por volquete",
                    labels={"ralenti_h": "Horas en ralentí", "id_vehiculo": "Vehículo"},
                )
                fig_idle.update_layout(height=max(350, len(temp_idle) * 38), margin=dict(l=10, r=10, t=60, b=10), yaxis_title="", xaxis_title="Horas en ralentí")
                st.plotly_chart(fig_idle, use_container_width=True)
            else:
                st.plotly_chart(plot_empty("Sin datos de ralentí."), use_container_width=True)
        else:
            st.plotly_chart(plot_empty("Sin datos de ralentí."), use_container_width=True)

    with col_d:
        # Comparación ralentí vs movimiento por vehículo (barras apiladas)
        if not vehicle_kpis.empty and {"ralenti_h", "movimiento_h"}.issubset(vehicle_kpis.columns):
            comp = vehicle_kpis[["id_vehiculo", "ralenti_h", "movimiento_h"]].dropna(subset=["ralenti_h", "movimiento_h"])
            if not comp.empty:
                comp_long = comp.melt(id_vars="id_vehiculo", value_vars=["ralenti_h", "movimiento_h"],
                                       var_name="tipo", value_name="horas")
                comp_long["tipo"] = comp_long["tipo"].map({"ralenti_h": "Ralentí", "movimiento_h": "Movimiento"})
                fig_comp = px.bar(
                    comp_long,
                    x="horas",
                    y="id_vehiculo",
                    color="tipo",
                    orientation="h",
                    barmode="stack",
                    title="Ralentí vs Movimiento por volquete",
                    labels={"horas": "Horas", "id_vehiculo": "Vehículo", "tipo": ""},
                    color_discrete_map={"Ralentí": "#F59E0B", "Movimiento": "#10B981"},
                )
                fig_comp.update_layout(height=max(350, len(comp) * 38), margin=dict(l=10, r=10, t=60, b=10), yaxis_title="", xaxis_title="Horas")
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.plotly_chart(plot_empty("Sin datos suficientes."), use_container_width=True)
        else:
            st.plotly_chart(plot_empty("Sin datos de ralentí y movimiento."), use_container_width=True)


# ---- 3. Comparación de Guardias ----
with tab_guardias:
    st.subheader("Comparación de Guardias: Día vs Noche")
    st.markdown(
        "<div class='section-note'>Guardia Día: 07:00–18:59 · Guardia Noche: 19:00–06:59. "
        "Los valores son promedios por vehículo dentro de cada guardia.</div>",
        unsafe_allow_html=True,
    )

    if guardia_kpis.empty:
        st.warning("No hay suficientes datos para comparar guardias.")
    else:
        colores_guardia = {"Día": "#F59E0B", "Noche": "#3B82F6"}

        g1, g2 = st.columns(2)

        with g1:
            if "horas_motor_h" in guardia_kpis.columns:
                fig_g_motor = px.bar(
                    guardia_kpis,
                    x="guardia",
                    y="horas_motor_h",
                    color="guardia",
                    title="Horas de motor promedio por guardia",
                    labels={"guardia": "Guardia", "horas_motor_h": "Horas de motor (prom.)"},
                    color_discrete_map=colores_guardia,
                )
                fig_g_motor.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), showlegend=False, xaxis_title="", yaxis_title="Horas de motor")
                st.plotly_chart(fig_g_motor, use_container_width=True)

        with g2:
            if "velocidad_kmh" in guardia_kpis.columns:
                fig_g_vel = px.bar(
                    guardia_kpis,
                    x="guardia",
                    y="velocidad_kmh",
                    color="guardia",
                    title="Velocidad promedio por guardia",
                    labels={"guardia": "Guardia", "velocidad_kmh": "km/h (prom.)"},
                    color_discrete_map=colores_guardia,
                )
                fig_g_vel.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), showlegend=False, xaxis_title="", yaxis_title="km/h")
                st.plotly_chart(fig_g_vel, use_container_width=True)

        g3, g4 = st.columns(2)

        with g3:
            if "consumo_L_hr" in guardia_kpis.columns:
                fig_g_cons = px.bar(
                    guardia_kpis,
                    x="guardia",
                    y="consumo_L_hr",
                    color="guardia",
                    title="Consumo promedio por guardia (L/hr)",
                    labels={"guardia": "Guardia", "consumo_L_hr": "L/hr (prom.)"},
                    color_discrete_map=colores_guardia,
                )
                fig_g_cons.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), showlegend=False, xaxis_title="", yaxis_title="L/hr")
                st.plotly_chart(fig_g_cons, use_container_width=True)

        with g4:
            if "ralenti_h" in guardia_kpis.columns:
                fig_g_idle = px.bar(
                    guardia_kpis,
                    x="guardia",
                    y="ralenti_h",
                    color="guardia",
                    title="Ralentí promedio por guardia",
                    labels={"guardia": "Guardia", "ralenti_h": "Horas en ralentí (prom.)"},
                    color_discrete_map=colores_guardia,
                )
                fig_g_idle.update_layout(height=380, margin=dict(l=10, r=10, t=60, b=10), showlegend=False, xaxis_title="", yaxis_title="Horas en ralentí")
                st.plotly_chart(fig_g_idle, use_container_width=True)

        st.subheader("Detalle numérico por guardia")
        if not guardia_kpis.empty:
            display_g = guardia_kpis.rename(columns={
                "guardia": "Guardia",
                "vehiculos": "Vehículos analizados",
                "velocidad_kmh": "Velocidad prom. (km/h)",
                "ralenti_h": "Ralentí prom. (h)",
                "horas_motor_h": "Horas motor prom. (h)",
                "consumo_L_hr": "Consumo prom. (L/hr)",
            }).round(2)
            st.dataframe(display_g, use_container_width=True, hide_index=True)


# ---- 4. Conclusiones ----
with tab_conclusiones:
    st.subheader("Conclusiones y Rankings")
    st.markdown(
        "<div class='section-note'>Top 5 por indicador clave. Las recomendaciones se generan automáticamente.</div>",
        unsafe_allow_html=True,
    )

    rank1, rank2, rank3 = st.columns(3)

    with rank1:
        st.markdown("##### Top 5 — Mayor consumo (L/hr)")
        if not vehicle_kpis.empty and "consumo_L_hr" in vehicle_kpis.columns:
            top_consumo = (
                vehicle_kpis[["id_vehiculo", "consumo_L_hr"]]
                .dropna()
                .sort_values("consumo_L_hr", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            top_consumo.index += 1
            top_consumo.columns = ["Vehículo", "L/hr"]
            st.dataframe(top_consumo.round(2), use_container_width=True)
        else:
            st.info("Sin datos de consumo.")

    with rank2:
        st.markdown("##### Top 5 — Mayor ralentí (h)")
        if not vehicle_kpis.empty and "ralenti_h" in vehicle_kpis.columns:
            top_ralenti = (
                vehicle_kpis[["id_vehiculo", "ralenti_h"]]
                .dropna()
                .sort_values("ralenti_h", ascending=False)
                .head(5)
                .reset_index(drop=True)
            )
            top_ralenti.index += 1
            top_ralenti.columns = ["Vehículo", "Horas ralentí"]
            st.dataframe(top_ralenti.round(2), use_container_width=True)
        else:
            st.info("Sin datos de ralentí.")

    with rank3:
        st.markdown("##### Top 5 — Menor velocidad (km/h)")
        if not vehicle_kpis.empty and "velocidad_promedio_kmh" in vehicle_kpis.columns:
            top_vel = (
                vehicle_kpis[["id_vehiculo", "velocidad_promedio_kmh"]]
                .dropna()
                .sort_values("velocidad_promedio_kmh", ascending=True)
                .head(5)
                .reset_index(drop=True)
            )
            top_vel.index += 1
            top_vel.columns = ["Vehículo", "km/h"]
            st.dataframe(top_vel.round(2), use_container_width=True)
        else:
            st.info("Sin datos de velocidad.")

    st.divider()
    st.markdown("#### Recomendaciones automáticas")

    recomendaciones = []

    if not vehicle_kpis.empty:
        # Mayor consumo
        if "consumo_L_hr" in vehicle_kpis.columns:
            top1_cons = vehicle_kpis.dropna(subset=["consumo_L_hr"]).sort_values("consumo_L_hr", ascending=False)
            if not top1_cons.empty:
                v = top1_cons.iloc[0]["id_vehiculo"]
                val = top1_cons.iloc[0]["consumo_L_hr"]
                recomendaciones.append(
                    f"🔴 El vehículo **{v}** presenta el mayor consumo de combustible "
                    f"({val:.2f} L/hr). Se recomienda revisar el estado mecánico del motor y los hábitos de conducción."
                )

        # Mayor ralentí
        if "ralenti_h" in vehicle_kpis.columns:
            top1_idle = vehicle_kpis.dropna(subset=["ralenti_h"]).sort_values("ralenti_h", ascending=False)
            if not top1_idle.empty:
                v = top1_idle.iloc[0]["id_vehiculo"]
                val = top1_idle.iloc[0]["ralenti_h"]
                recomendaciones.append(
                    f"🟡 El vehículo **{v}** registra el mayor tiempo en ralentí "
                    f"({val:.1f} h), lo que podría indicar tiempos de espera excesivos u oportunidades de mejora operativa."
                )

        # Menor velocidad
        if "velocidad_promedio_kmh" in vehicle_kpis.columns:
            top1_vel = vehicle_kpis.dropna(subset=["velocidad_promedio_kmh"]).sort_values("velocidad_promedio_kmh", ascending=True)
            if not top1_vel.empty:
                v = top1_vel.iloc[0]["id_vehiculo"]
                val = top1_vel.iloc[0]["velocidad_promedio_kmh"]
                recomendaciones.append(
                    f"🟠 El vehículo **{v}** presenta la menor velocidad promedio "
                    f"({val:.1f} km/h). Evaluar si se debe a la ruta asignada o a condiciones mecánicas."
                )

    # Comparación de guardias
    if not guardia_kpis.empty and "ralenti_h" in guardia_kpis.columns and len(guardia_kpis) == 2:
        g_sorted = guardia_kpis.sort_values("ralenti_h", ascending=False)
        guardia_mayor = g_sorted.iloc[0]["guardia"]
        val_mayor = g_sorted.iloc[0]["ralenti_h"]
        guardia_menor = g_sorted.iloc[1]["guardia"]
        val_menor = g_sorted.iloc[1]["ralenti_h"]
        if val_mayor > val_menor * 1.1:
            recomendaciones.append(
                f"🔵 La guardia **{guardia_mayor}** presenta en promedio más horas de ralentí "
                f"({val_mayor:.1f} h) que la guardia {guardia_menor} ({val_menor:.1f} h). "
                f"Se recomienda revisar los procedimientos de espera durante la guardia {guardia_mayor}."
            )

    if recomendaciones:
        for r in recomendaciones:
            st.markdown(r)
    else:
        st.info("No hay suficientes datos para generar recomendaciones automáticas.")

st.caption("Dashboard ejecutivo JRC · Los KPIs de consumo y ralentí se calculan como delta de acumulados por vehículo en el periodo seleccionado.")
