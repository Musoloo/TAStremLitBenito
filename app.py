from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "streamlit_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "streamlit_matplotlib_cache"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA


# =========================
# KONFIGURASI UTAMA
# =========================

DATA_FILE = Path(__file__).with_name("edited data TA.xlsx")

DATE_COLUMN = "Tanggal Job"
VALUE_COLUMN = "Jumlah Cylinder"

ARIMA_ORDER = (3, 0, 3)
MODEL_NAME = "ARIMA(3,0,3)"

PRIMARY = "#1a73e8"
SECONDARY = "#00a86b"
AMBER = "#fbbc04"
CORAL = "#e8710a"
TEXT = "#1f2937"
MUTED = "#64748b"
SURFACE = "#ffffff"
BG = "#f6f8fc"


st.set_page_config(
    page_title="Dashboard Forecast Permintaan Cylinder",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================
# CSS DASHBOARD
# =========================

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {PRIMARY};
            --secondary: {SECONDARY};
            --text: {TEXT};
            --muted: {MUTED};
            --surface: {SURFACE};
            --bg: {BG};
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(26,115,232,.08), transparent 30rem),
                linear-gradient(180deg, #f9fbff 0%, var(--bg) 42%, #ffffff 100%);
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid #e6eaf2;
        }}

        [data-testid="stHeader"] {{
            background: rgba(246,248,252,.75);
            backdrop-filter: blur(10px);
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }}

        .hero {{
            padding: 1.5rem 1.5rem 1.2rem;
            border: 1px solid #e5eaf3;
            border-radius: 8px;
            background: rgba(255,255,255,.92);
            box-shadow: 0 8px 28px rgba(15, 23, 42, .06);
            margin-bottom: 1rem;
        }}

        .hero h1 {{
            color: var(--text);
            font-size: clamp(1.65rem, 2.8vw, 2.55rem);
            line-height: 1.12;
            margin: 0 0 .45rem;
            letter-spacing: 0;
        }}

        .hero p {{
            color: var(--muted);
            font-size: 1rem;
            margin: 0;
        }}

        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: .9rem;
            margin: 1rem 0 1.1rem;
        }}

        .kpi-card {{
            background: var(--surface);
            border: 1px solid #e5eaf3;
            border-radius: 8px;
            padding: 1rem;
            min-height: 112px;
            box-shadow: 0 8px 24px rgba(15, 23, 42, .055);
        }}

        .kpi-label {{
            color: var(--muted);
            font-size: .82rem;
            line-height: 1.2;
            margin-bottom: .55rem;
        }}

        .kpi-value {{
            color: var(--text);
            font-size: clamp(1.4rem, 2.2vw, 2rem);
            font-weight: 760;
            line-height: 1.05;
            overflow-wrap: anywhere;
        }}

        .kpi-help {{
            color: var(--muted);
            font-size: .78rem;
            margin-top: .45rem;
        }}

        .analysis-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: .75rem;
            margin: .55rem 0 .85rem;
        }}

        .analysis-card {{
            background: #ffffff;
            border: 1px solid #e5eaf3;
            border-radius: 8px;
            padding: .85rem .9rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, .045);
        }}

        .analysis-card.peak {{
            border-left: 4px solid var(--secondary);
        }}

        .analysis-label {{
            color: var(--muted);
            font-size: .78rem;
            line-height: 1.2;
            margin-bottom: .35rem;
        }}

        .analysis-value {{
            color: var(--text);
            font-size: 1.2rem;
            font-weight: 760;
            line-height: 1.1;
            overflow-wrap: anywhere;
        }}

        .analysis-help {{
            color: var(--muted);
            font-size: .74rem;
            margin-top: .35rem;
        }}

        .section-title {{
            color: var(--text);
            font-size: 1.12rem;
            font-weight: 760;
            margin: .4rem 0 .25rem;
        }}

        .stDataFrame, [data-testid="stTable"] {{
            border: 1px solid #e5eaf3;
            border-radius: 8px;
            overflow: hidden;
        }}

        @media (max-width: 1100px) {{
            .kpi-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
            .analysis-grid {{
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }}
        }}

        @media (max-width: 640px) {{
            .block-container {{
                padding-left: .9rem;
                padding-right: .9rem;
            }}
            .hero {{
                padding: 1rem;
            }}
            .kpi-grid {{
                grid-template-columns: 1fr;
            }}
            .analysis-grid {{
                grid-template-columns: 1fr;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# LOAD DAN OLAH DATA
# =========================

@st.cache_data(show_spinner=False)
def load_workbook(source) -> pd.DataFrame:
    df = pd.read_excel(source)

    missing = {DATE_COLUMN, VALUE_COLUMN}.difference(df.columns)
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(sorted(missing))
            + f". Kolom yang tersedia: {', '.join(map(str, df.columns))}"
        )

    data = df[[DATE_COLUMN, VALUE_COLUMN]].copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data[VALUE_COLUMN] = pd.to_numeric(data[VALUE_COLUMN], errors="coerce")

    data = data.dropna(subset=[DATE_COLUMN, VALUE_COLUMN])
    data[VALUE_COLUMN] = data[VALUE_COLUMN].clip(lower=0)

    return data.sort_values(DATE_COLUMN).reset_index(drop=True)


def aggregate_series(
    df: pd.DataFrame,
    start_date,
    end_date,
    frequency: str,
) -> tuple[pd.DataFrame, pd.Series]:
    mask = df[DATE_COLUMN].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    filtered = df.loc[mask].copy()

    daily = (
        filtered.groupby(DATE_COLUMN, as_index=True)[VALUE_COLUMN]
        .sum()
        .sort_index()
        .rename("Permintaan")
    )

    if daily.empty:
        return filtered, pd.Series(dtype=float, name="Permintaan")

    daily = daily.asfreq("D", fill_value=0)
    series = daily.resample(frequency).sum().astype(float)
    series = series.ffill()
    series.name = "Permintaan"

    return filtered, series


# =========================
# MODEL ARIMA
# =========================

def evaluate_forecast(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    actual = y_true.astype(float)
    pred = pd.Series(y_pred, index=actual.index).astype(float)

    mae = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))

    denominator = actual.replace(0, np.nan)
    mape = float((np.abs((actual - pred) / denominator).dropna().mean() * 100))

    if np.isnan(mape):
        mape = 0.0

    return mae, rmse, mape


@st.cache_data(show_spinner=False)
def run_arima_model(
    series: pd.Series,
    forecast_steps: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, float, float, float, str]:
    if len(series) < 10:
        raise ValueError(
            "Data terlalu sedikit untuk menjalankan ARIMA. "
            "Gunakan rentang tanggal yang lebih panjang."
        )

    train_size = max(1, int(len(series) * 0.8))
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    if test.empty:
        raise ValueError("Data test kosong. Tambahkan rentang tanggal atau kurangi filter.")

    note = ""

    try:
        test_fit = ARIMA(train, order=ARIMA_ORDER).fit()
        test_forecast = test_fit.forecast(steps=len(test))
        test_forecast = pd.Series(
            test_forecast.to_numpy(),
            index=test.index,
            name="Prediksi ARIMA"
        )
        test_forecast = test_forecast.clip(lower=0)

        mae, rmse, mape = evaluate_forecast(test, test_forecast)

    except Exception as exc:
        raise ValueError(f"Model {MODEL_NAME} gagal menghitung data testing: {exc}")

    try:
        final_fit = ARIMA(series, order=ARIMA_ORDER).fit()
        future = final_fit.forecast(steps=forecast_steps)
        future = pd.Series(
            future.to_numpy(),
            index=future.index,
            name="Forecast ARIMA"
        )
        future = future.clip(lower=0)

    except Exception as exc:
        last_value = float(series.iloc[-1])
        freq = series.index.freqstr or "W"
        future_index = pd.date_range(
            start=series.index[-1] + pd.tseries.frequencies.to_offset(freq),
            periods=forecast_steps,
            freq=freq,
        )
        future = pd.Series(
            [last_value] * forecast_steps,
            index=future_index,
            name="Forecast ARIMA"
        )
        note = (
            f"Forecast masa depan memakai nilai terakhir karena model {MODEL_NAME} "
            f"gagal menghitung forecast lanjutan: {exc}"
        )

    return train, test, test_forecast, future, mae, rmse, mape, note


# =========================
# FORMAT DAN KOMPONEN UI
# =========================

def format_number(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{decimals}f}".replace(",", ".")


def kpi_card(label: str, value: str, help_text: str) -> str:
    return (
        '<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-help">{help_text}</div>'
        "</div>"
    )


def render_kpis(
    df: pd.DataFrame,
    series: pd.Series,
    mae: float,
    rmse: float,
    mape: float,
) -> None:
    total = df[VALUE_COLUMN].sum()
    avg = series.mean()
    latest_date = series.index[-1].strftime("%d %b %Y")
    latest_value = series.iloc[-1]

    cards = [
        kpi_card("Total cylinder", format_number(total), f"{len(df):,} transaksi".replace(",", ".")),
        kpi_card("Rata-rata per periode", format_number(avg, 1), "berdasarkan filter aktif"),
        kpi_card("Model yang digunakan", MODEL_NAME, "model utama penelitian"),
        kpi_card("MAPE", f"{format_number(mape, 2)}%", f"RMSE: {format_number(rmse, 2)} | MAE: {format_number(mae, 2)}"),
    ]

    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)

    st.caption(
        f"Periode terakhir pada data aktual adalah {latest_date} "
        f"dengan jumlah permintaan {format_number(latest_value)} cylinder."
    )


def analysis_card(label: str, value: str, help_text: str, highlight: bool = False) -> str:
    peak_class = " peak" if highlight else ""
    return (
        f'<div class="analysis-card{peak_class}">'
        f'<div class="analysis-label">{label}</div>'
        f'<div class="analysis-value">{value}</div>'
        f'<div class="analysis-help">{help_text}</div>'
        "</div>"
    )


def base_chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=50, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Roboto, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#d8dee9")
    fig.update_yaxes(gridcolor="#e8edf5", zerolinecolor="#e8edf5")
    return fig


# =========================
# GRAFIK
# =========================

def actual_forecast_chart(series: pd.Series, future: pd.Series) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines+markers",
            name="Data Aktual",
            line=dict(color=PRIMARY, width=3),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future.index,
            y=future.values,
            mode="lines+markers",
            name=f"Forecast {MODEL_NAME}",
            line=dict(color=SECONDARY, width=3, dash="dash"),
            marker=dict(size=7),
        )
    )

    fig.update_layout(title=f"Trend Permintaan Aktual dan Forecast {MODEL_NAME}")
    fig.update_yaxes(title="Jumlah Cylinder")

    return base_chart_layout(fig, 460)


def actual_vs_arima_chart(test: pd.Series, test_forecast: pd.Series) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines+markers",
            name="Aktual",
            line=dict(color=TEXT, width=3),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=test_forecast.index,
            y=test_forecast.values,
            mode="lines+markers",
            name=f"Prediksi {MODEL_NAME}",
            line=dict(color=PRIMARY, width=3, dash="dash"),
            marker=dict(size=7),
        )
    )

    fig.update_layout(title=f"Perbandingan Aktual dan Prediksi {MODEL_NAME} pada Data Testing")
    fig.update_yaxes(title="Jumlah Cylinder")

    return base_chart_layout(fig, 430)


def four_week_peak_chart(future: pd.Series) -> go.Figure:
    horizon = future.head(4).copy()
    x_positions = list(range(1, len(horizon) + 1))
    labels = [f"W{idx}<br>{date.strftime('%d %b')}" for idx, date in enumerate(horizon.index, start=1)]

    peak_position = int(np.argmax(horizon.values)) + 1
    peak_value = float(horizon.max())

    y_min = max(0, float(horizon.min()) * 0.88)
    y_max = float(horizon.max()) * 1.18

    if y_max == y_min:
        y_max = y_min + 1

    fig = go.Figure()

    fig.add_vrect(
        x0=peak_position - 0.5,
        x1=peak_position + 0.5,
        fillcolor="#fff1a8",
        opacity=0.72,
        line_width=0,
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=x_positions,
            y=horizon.values,
            mode="lines+markers",
            name=f"Forecast {MODEL_NAME}",
            line=dict(color="#c92535", width=4, shape="spline"),
            marker=dict(size=9, color="#c92535", line=dict(color="#ffffff", width=2)),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[peak_position],
            y=[peak_value],
            mode="markers",
            name="Peak Demand",
            marker=dict(size=15, color="#fbbc04", line=dict(color="#c92535", width=3)),
            showlegend=False,
            hovertemplate="Peak Demand<br>%{y:,.0f} cylinder<extra></extra>",
        )
    )

    fig.add_annotation(
        x=peak_position,
        y=y_max * 0.965,
        text="<b>Peak<br>Demand</b>",
        showarrow=False,
        font=dict(size=18, color="#4b3b06"),
        align="center",
        bgcolor="rgba(255, 241, 168, 0.72)",
        borderpad=6,
    )

    fig.update_layout(
        title=dict(
            text="<b>Forecast Permintaan Cylinder 4 Periode ke Depan</b><br><sup>ARIMA(3,0,3)</sup>",
            x=0,
            xanchor="left",
            font=dict(size=21, color=TEXT),
        ),
        height=420,
        margin=dict(l=16, r=16, t=82, b=58),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        font=dict(color=TEXT, family="Inter, Roboto, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        hovermode="x unified",
    )

    fig.update_xaxes(
        tickmode="array",
        tickvals=x_positions,
        ticktext=labels,
        range=[0.65, len(horizon) + 0.35],
        showgrid=False,
        linecolor="#d8dee9",
        title=None,
    )

    fig.update_yaxes(
        title="Jumlah Cylinder",
        range=[y_min, y_max],
        gridcolor="#e8edf5",
        zerolinecolor="#e8edf5",
    )

    return fig


# =========================
# TABEL DAN ANALISIS
# =========================

def evaluation_table(mae: float, rmse: float, mape: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Model": [MODEL_NAME],
            "MAE": [mae],
            "RMSE": [rmse],
            "MAPE (%)": [mape],
        }
    )


def forecast_table(future: pd.Series) -> pd.DataFrame:
    table = future.reset_index()
    table.columns = ["Periode", "Forecast Cylinder"]
    table["Forecast Cylinder"] = table["Forecast Cylinder"].round(0)
    return table


def actual_prediction_table(test: pd.Series, test_forecast: pd.Series) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Periode": test.index,
            "Aktual": test.values,
            "Prediksi ARIMA": test_forecast.values,
        }
    )

    table["Selisih"] = table["Aktual"] - table["Prediksi ARIMA"]
    table["Absolute Error"] = table["Selisih"].abs()

    return table


def four_week_analysis(future: pd.Series) -> tuple[pd.DataFrame, str]:
    horizon = future.head(4).copy()
    peak_date = horizon.idxmax()
    peak_value = float(horizon.max())
    total = float(horizon.sum())
    average = float(horizon.mean())
    delta = float(horizon.iloc[-1] - horizon.iloc[0])

    direction = "naik" if delta > 0 else "turun" if delta < 0 else "stabil"

    cards = [
        analysis_card("Total forecast 4 periode", format_number(total), "akumulasi kebutuhan forecast"),
        analysis_card("Rata-rata per periode", format_number(average, 1), "rata-rata dari 4 periode forecast"),
        analysis_card(
            "Peak demand forecast",
            format_number(peak_value),
            peak_date.strftime("%d %b %Y"),
            highlight=True,
        ),
    ]

    table = horizon.reset_index()
    table.columns = ["Periode", "Forecast Cylinder"]
    table.insert(0, "Periode Forecast", [f"Periode {idx}" for idx in range(1, len(table) + 1)])
    table["Status"] = np.where(table["Periode"] == peak_date, "Peak demand", "Normal")

    insight = (
        f"Berdasarkan hasil forecast {MODEL_NAME}, peak demand dari 4 periode ke depan "
        f"diperkirakan terjadi pada {peak_date.strftime('%d %b %Y')} dengan estimasi "
        f"{format_number(peak_value)} cylinder. Tren forecast terlihat {direction} sebesar "
        f"{format_number(abs(delta), 1)} cylinder dari periode pertama ke periode keempat."
    )

    summary_html = f"<div class='analysis-grid'>{''.join(cards)}</div><p>{insight}</p>"

    return table, summary_html


# =========================
# MAIN APP
# =========================

def main() -> None:
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>Dashboard Forecast Permintaan Cylinder</h1>
            <p>Visualisasi hasil peramalan permintaan menggunakan model ARIMA(3,0,3).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Kontrol Dashboard")

        uploaded_file = st.file_uploader(
            "Gunakan file Excel lain",
            type=["xlsx", "xls"],
            help="File harus memiliki kolom Tanggal Job dan Jumlah Cylinder."
        )

        source = uploaded_file if uploaded_file is not None else DATA_FILE

        try:
            df = load_workbook(source)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

        min_date = df[DATE_COLUMN].min().date()
        max_date = df[DATE_COLUMN].max().date()

        selected_dates = st.date_input(
            "Rentang tanggal",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if not isinstance(selected_dates, tuple) or len(selected_dates) != 2:
            st.info("Pilih tanggal mulai dan tanggal akhir.")
            st.stop()

        frequency_label = st.segmented_control(
            "Agregasi",
            options=["Mingguan", "Bulanan"],
            default="Mingguan",
        )

        frequency = "W" if frequency_label == "Mingguan" else "MS"

        forecast_steps = st.slider(
            "Jumlah periode forecast",
            min_value=4,
            max_value=12,
            value=4,
        )

    filtered_df, series = aggregate_series(
        df,
        selected_dates[0],
        selected_dates[1],
        frequency,
    )

    if filtered_df.empty or series.empty:
        st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
        st.stop()

    with st.spinner(f"Menghitung model {MODEL_NAME}..."):
        try:
            train, test, test_forecast, future, mae, rmse, mape, arima_note = run_arima_model(
                series,
                forecast_steps,
            )
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    render_kpis(filtered_df, series, mae, rmse, mape)

    if arima_note:
        st.info(arima_note)

    left, right = st.columns([1.45, 1], gap="large")

    with left:
        st.markdown("<div class='section-title'>Forecast Permintaan</div>", unsafe_allow_html=True)
        st.plotly_chart(actual_forecast_chart(series, future), width="stretch")

    with right:
        st.markdown("<div class='section-title'>Evaluasi Model ARIMA</div>", unsafe_allow_html=True)

        eval_df = evaluation_table(mae, rmse, mape)
        st.dataframe(
            eval_df.style.format({
                "MAE": "{:,.2f}",
                "RMSE": "{:,.2f}",
                "MAPE (%)": "{:,.2f}",
            }),
            width="stretch",
            hide_index=True,
        )

        st.caption(
            "Evaluasi dilakukan dengan membandingkan data aktual dan prediksi ARIMA "
            "pada data testing sebesar 20% dari total data."
        )

        st.markdown("<div class='section-title'>Tabel Forecast</div>", unsafe_allow_html=True)

        future_df = forecast_table(future)
        st.dataframe(
            future_df.style.format({"Forecast Cylinder": "{:,.0f}"}),
            width="stretch",
            hide_index=True,
        )

    st.markdown("<div class='section-title'>Analisa Forecast 4 Periode ke Depan</div>", unsafe_allow_html=True)

    four_week_table, four_week_summary = four_week_analysis(future)
    st.markdown(four_week_summary, unsafe_allow_html=True)

    st.plotly_chart(four_week_peak_chart(future), width="stretch")

    st.dataframe(
        four_week_table.style.format({"Forecast Cylinder": "{:,.0f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown("<div class='section-title'>Aktual vs Prediksi ARIMA pada Data Testing</div>", unsafe_allow_html=True)

    st.plotly_chart(actual_vs_arima_chart(test, test_forecast), width="stretch")

    prediction_df = actual_prediction_table(test, test_forecast)
    st.dataframe(
        prediction_df.style.format({
            "Aktual": "{:,.0f}",
            "Prediksi ARIMA": "{:,.0f}",
            "Selisih": "{:,.2f}",
            "Absolute Error": "{:,.2f}",
        }),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Lihat data olahan"):
        tab_daily, tab_periodic, tab_raw = st.tabs(["Harian", "Agregasi", "Transaksi"])

        daily_table = (
            filtered_df.groupby(DATE_COLUMN, as_index=False)[VALUE_COLUMN]
            .sum()
            .sort_values(DATE_COLUMN)
        )

        periodic_table = series.reset_index()
        periodic_table.columns = ["Periode", "Jumlah Cylinder"]

        tab_daily.dataframe(daily_table, width="stretch", hide_index=True)
        tab_periodic.dataframe(periodic_table, width="stretch", hide_index=True)
        tab_raw.dataframe(filtered_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
