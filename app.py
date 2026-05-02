from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "streamlit_cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "streamlit_matplotlib_cache"))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA


DATA_FILE = Path(__file__).with_name("edited data TA benito.xlsx")
DATE_COLUMN = "Tanggal Job"
VALUE_COLUMN = "Jumlah Cylinder"

PRIMARY = "#1a73e8"
SECONDARY = "#00a86b"
AMBER = "#fbbc04"
CORAL = "#e8710a"
TEXT = "#1f2937"
MUTED = "#64748b"
SURFACE = "#ffffff"
BG = "#f6f8fc"


@dataclass(frozen=True)
class ModelResult:
    name: str
    predictions: pd.Series
    mae: float
    rmse: float
    mape: float


st.set_page_config(
    page_title="Dashboard Forecast Permintaan Cylinder",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


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

        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid #e5eaf3;
            border-radius: 8px;
            padding: .85rem 1rem;
            box-shadow: 0 6px 20px rgba(15, 23, 42, .045);
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


def aggregate_series(df: pd.DataFrame, start_date, end_date, frequency: str) -> tuple[pd.DataFrame, pd.Series]:
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


def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    pred = pd.Series(y_pred, index=y_true.index).astype(float)
    actual = y_true.astype(float)
    mae = mean_absolute_error(actual, pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    denominator = actual.replace(0, np.nan)
    mape = float((np.abs((actual - pred) / denominator).dropna().mean() * 100))
    if np.isnan(mape):
        mape = 0.0
    return float(mae), rmse, mape


@st.cache_data(show_spinner=False)
def run_forecast_models(
    series: pd.Series,
    moving_average_window: int,
    forecast_steps: int,
) -> tuple[pd.Series, pd.Series, list[ModelResult], pd.Series, str]:
    if len(series) < 8:
        raise ValueError("Data terlalu sedikit untuk train/test forecasting. Gunakan rentang tanggal yang lebih panjang.")

    train_size = max(1, int(len(series) * 0.8))
    train = series.iloc[:train_size]
    test = series.iloc[train_size:]

    if test.empty:
        raise ValueError("Data test kosong. Tambahkan rentang tanggal atau kurangi filter.")

    history = train.astype(float).tolist()
    naive_values = []
    for value in test:
        naive_values.append(history[-1])
        history.append(float(value))
    naive_pred = pd.Series(naive_values, index=test.index, name="Naive")
    naive_metrics = evaluate_model(test, naive_pred)

    history = train.astype(float).tolist()
    ma_values = []
    window = max(1, min(moving_average_window, len(history)))
    for value in test:
        ma_values.append(float(np.mean(history[-window:])))
        history.append(float(value))
    ma_pred = pd.Series(ma_values, index=test.index, name="Moving Average")
    ma_metrics = evaluate_model(test, ma_pred)

    results = [
        ModelResult("Naive", naive_pred, *naive_metrics),
        ModelResult("Moving Average", ma_pred, *ma_metrics),
    ]

    arima_note = ""
    try:
        arima_fit = ARIMA(train, order=(1, 1, 1)).fit()
        arima_pred = arima_fit.forecast(steps=len(test))
        arima_pred = pd.Series(arima_pred.to_numpy(), index=test.index, name="ARIMA")
        arima_metrics = evaluate_model(test, arima_pred)
        results.append(ModelResult("ARIMA", arima_pred, *arima_metrics))
    except Exception as exc:  # pragma: no cover - surfaced in UI
        arima_note = f"ARIMA tidak dapat dihitung untuk rentang ini: {exc}"

    try:
        final_fit = ARIMA(series, order=(1, 1, 1)).fit()
        future = final_fit.forecast(steps=forecast_steps)
        future = pd.Series(future.to_numpy(), index=future.index, name="Forecast")
        future = future.clip(lower=0)
    except Exception:
        last_value = float(series.iloc[-1])
        future_index = pd.date_range(
            start=series.index[-1] + pd.tseries.frequencies.to_offset(series.index.freqstr or "W"),
            periods=forecast_steps,
            freq=series.index.freqstr or "W",
        )
        future = pd.Series([last_value] * forecast_steps, index=future_index, name="Forecast")
        arima_note = "Forecast masa depan memakai fallback nilai terakhir karena ARIMA gagal pada rentang ini."

    return train, test, results, future, arima_note


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


def render_kpis(df: pd.DataFrame, series: pd.Series, best_model: str) -> None:
    total = df[VALUE_COLUMN].sum()
    avg = series.mean()
    peak_date = series.idxmax().strftime("%d %b %Y")
    peak_value = series.max()
    latest_date = series.index[-1].strftime("%d %b %Y")
    latest_value = series.iloc[-1]

    cards = [
        kpi_card("Total cylinder", format_number(total), f"{len(df):,} transaksi".replace(",", ".")),
        kpi_card("Rata-rata per periode", format_number(avg, 1), "berdasarkan filter aktif"),
        kpi_card("Puncak permintaan", format_number(peak_value), peak_date),
        kpi_card("Model terbaik", best_model, f"periode terakhir {latest_date}: {format_number(latest_value)}"),
    ]
    st.markdown(f"<div class='kpi-grid'>{''.join(cards)}</div>", unsafe_allow_html=True)


def analysis_card(label: str, value: str, help_text: str, highlight: bool = False) -> str:
    peak_class = " peak" if highlight else ""
    return (
        f'<div class="analysis-card{peak_class}">'
        f'<div class="analysis-label">{label}</div>'
        f'<div class="analysis-value">{value}</div>'
        f'<div class="analysis-help">{help_text}</div>'
        "</div>"
    )


def four_week_analysis(future: pd.Series) -> tuple[pd.DataFrame, str]:
    horizon = future.head(4).copy()
    horizon.name = "Forecast Cylinder"
    peak_date = horizon.idxmax()
    peak_value = float(horizon.max())
    total = float(horizon.sum())
    average = float(horizon.mean())
    delta = float(horizon.iloc[-1] - horizon.iloc[0])
    direction = "naik" if delta > 0 else "turun" if delta < 0 else "stabil"

    cards = [
        analysis_card("Total forecast 4 minggu", format_number(total), "akumulasi kebutuhan forecast"),
        analysis_card("Rata-rata per minggu", format_number(average, 1), "rata-rata dari 4 minggu forecast"),
        analysis_card(
            "Peak demand 4 minggu",
            format_number(peak_value),
            peak_date.strftime("%d %b %Y"),
            highlight=True,
        ),
    ]

    table = horizon.reset_index()
    table.columns = ["Periode", "Forecast Cylinder"]
    table.insert(0, "Minggu", [f"Minggu {idx}" for idx in range(1, len(table) + 1)])
    table["Status"] = np.where(table["Periode"] == peak_date, "Peak demand", "Normal")

    insight = (
        f"Peak demand dari 4 minggu forecast ada pada {peak_date.strftime('%d %b %Y')} "
        f"dengan estimasi {format_number(peak_value)} cylinder. Tren 4 minggu terlihat {direction} "
        f"sebesar {format_number(abs(delta), 1)} cylinder dari minggu pertama ke minggu keempat."
    )
    return table, f"<div class='analysis-grid'>{''.join(cards)}</div><p>{insight}</p>"


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
            name="ARIMA Forecast",
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
            text="<b>Request Order Cylinder<br>Forecast</b><br><sup>Weekly Demand Projection</sup>",
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


def base_chart_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=42, b=16),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Inter, Roboto, Arial, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, linecolor="#d8dee9")
    fig.update_yaxes(gridcolor="#e8edf5", zerolinecolor="#e8edf5")
    return fig


def trend_chart(series: pd.Series, future: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines+markers",
            name="Actual",
            line=dict(color=PRIMARY, width=3),
            marker=dict(size=6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future.index,
            y=future.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(color=SECONDARY, width=3, dash="dash"),
            marker=dict(size=7),
        )
    )
    fig.update_layout(title="Trend Permintaan dan Forecast ke Depan")
    return base_chart_layout(fig, 460)


def comparison_chart(test: pd.Series, results: list[ModelResult]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test.values,
            mode="lines+markers",
            name="Actual",
            line=dict(color=TEXT, width=3),
        )
    )
    colors = {"Naive": AMBER, "Moving Average": CORAL, "ARIMA": PRIMARY}
    for result in results:
        fig.add_trace(
            go.Scatter(
                x=result.predictions.index,
                y=result.predictions.values,
                mode="lines+markers",
                name=result.name,
                line=dict(color=colors.get(result.name, SECONDARY), width=2, dash="dash"),
            )
        )
    fig.update_layout(title="Perbandingan Model pada Data Test")
    return base_chart_layout(fig, 420)


def distribution_chart(series: pd.Series) -> go.Figure:
    fig = px.histogram(
        series.reset_index(name="Permintaan"),
        x="Permintaan",
        nbins=12,
        color_discrete_sequence=[PRIMARY],
    )
    fig.update_traces(marker_line_width=0)
    fig.update_layout(title="Distribusi Permintaan per Periode", bargap=0.08)
    return base_chart_layout(fig, 360)


def metrics_table(results: list[ModelResult]) -> pd.DataFrame:
    table = pd.DataFrame(
        {
            "Model": [result.name for result in results],
            "MAE": [result.mae for result in results],
            "RMSE": [result.rmse for result in results],
            "MAPE (%)": [result.mape for result in results],
        }
    )
    return table.sort_values("RMSE", ascending=True).reset_index(drop=True)


def descriptive_table(series: pd.Series) -> pd.DataFrame:
    desc = series.describe().rename(
        {
            "count": "Jumlah Data",
            "mean": "Rata-rata",
            "std": "Standar Deviasi",
            "min": "Minimum",
            "25%": "Kuartil 1",
            "50%": "Median",
            "75%": "Kuartil 3",
            "max": "Maksimum",
        }
    )
    return desc.to_frame("Nilai")


def main() -> None:
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>Dashboard Forecast Permintaan Cylinder</h1>
            <p>Monitoring tren permintaan, evaluasi model forecasting, dan proyeksi kebutuhan beberapa periode ke depan.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Kontrol Dashboard")
        uploaded_file = st.file_uploader("Gunakan file Excel lain", type=["xlsx", "xls"])
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

        forecast_steps = st.slider("Jumlah periode forecast", 4, 12, 4)
        ma_window = st.slider("Window Moving Average", 2, 8, 3)

    filtered_df, series = aggregate_series(df, selected_dates[0], selected_dates[1], frequency)
    if filtered_df.empty or series.empty:
        st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
        st.stop()

    with st.spinner("Menghitung model forecasting..."):
        try:
            train, test, results, future, arima_note = run_forecast_models(series, ma_window, forecast_steps)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    ranking = metrics_table(results)
    best_model = ranking.iloc[0]["Model"]
    render_kpis(filtered_df, series, best_model)

    if arima_note:
        st.info(arima_note)

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown("<div class='section-title'>Forecast</div>", unsafe_allow_html=True)
        st.plotly_chart(trend_chart(series, future), width="stretch")
    with right:
        st.markdown("<div class='section-title'>Evaluasi Model</div>", unsafe_allow_html=True)
        st.dataframe(
            ranking.style.format({"MAE": "{:,.2f}", "RMSE": "{:,.2f}", "MAPE (%)": "{:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption("Model terbaik dipilih dari RMSE terkecil pada data test 20%.")

        future_table = future.reset_index()
        future_table.columns = ["Periode", "Forecast Cylinder"]
        st.dataframe(
            future_table.style.format({"Forecast Cylinder": "{:,.0f}"}),
            width="stretch",
            hide_index=True,
        )

    st.markdown("<div class='section-title'>Analisa 4 Minggu ke Depan</div>", unsafe_allow_html=True)
    four_week_table, four_week_summary = four_week_analysis(future)
    st.markdown(four_week_summary, unsafe_allow_html=True)
    st.plotly_chart(four_week_peak_chart(future), width="stretch")
    st.dataframe(
        four_week_table.style.format({"Forecast Cylinder": "{:,.0f}"}),
        width="stretch",
        hide_index=True,
    )

    st.markdown("<div class='section-title'>Perbandingan Aktual vs Prediksi</div>", unsafe_allow_html=True)
    st.plotly_chart(comparison_chart(test, results), width="stretch")

    col_a, col_b = st.columns([1, 1], gap="large")
    with col_a:
        st.markdown("<div class='section-title'>Distribusi Data</div>", unsafe_allow_html=True)
        st.plotly_chart(distribution_chart(series), width="stretch")
    with col_b:
        st.markdown("<div class='section-title'>Statistik Deskriptif</div>", unsafe_allow_html=True)
        st.dataframe(
            descriptive_table(series).style.format({"Nilai": "{:,.2f}"}),
            width="stretch",
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
