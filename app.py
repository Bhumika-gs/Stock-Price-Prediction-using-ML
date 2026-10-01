import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Stock Price Prediction",
    page_icon="📈",
    layout="wide"
)


# ============================================================
# COLORS
# ============================================================

FOREST_GREEN = "#174D3A"
DARK_GREEN = "#123C2D"
SAGE_GREEN = "#A9C3A5"
LIGHT_SAGE = "#E7EFE4"
CREAM = "#FAF9F4"
WHITE = "#FFFFFF"
CORAL = "#F28C78"
TEXT_GREEN = "#244A3A"
LIGHT_BORDER = "#D7E2D4"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
    <style>

    /* Main application */
    .stApp {{
        background-color: {CREAM};
        color: {TEXT_GREEN};
    }}

    /* Remove unnecessary top spacing */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {FOREST_GREEN};
    }}

    section[data-testid="stSidebar"] * {{
        color: white;
    }}

    /* Header */
    .header {{
        background-color: {LIGHT_SAGE};
        border: 1px solid {SAGE_GREEN};
        border-radius: 18px;
        padding: 30px 35px;
        margin-bottom: 25px;
    }}

    .header h1 {{
        color: {FOREST_GREEN};
        font-size: 48px;
        font-weight: 800;
        margin: 0;
        line-height: 1.15;
    }}

    .header p {{
        color: #58705F;
        font-size: 18px;
        margin-top: 10px;
        margin-bottom: 0;
    }}

    /* Section headings */
    .section-title {{
        color: {FOREST_GREEN};
        font-size: 24px;
        font-weight: 700;
        margin-top: 22px;
        margin-bottom: 12px;
    }}

    /* Cards */
    .card {{
        background-color: {WHITE};
        border: 1px solid {LIGHT_BORDER};
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(23, 77, 58, 0.06);
    }}

    /* Prediction area */
    .prediction-box {{
        background-color: #F5F2EB;
        border: 1px solid #E5DCCF;
        border-radius: 14px;
        padding: 18px;
        margin-top: 15px;
    }}

    /* Metric */
    div[data-testid="metric-container"] {{
        background-color: {WHITE};
        border: 1px solid {LIGHT_BORDER};
        border-radius: 12px;
        padding: 18px;
    }}

    div[data-testid="metric-container"] label {{
        color: #65776B !important;
    }}

    div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
        color: {FOREST_GREEN} !important;
        font-weight: 700;
    }}

    /* Button */
    .stButton > button {{
        background-color: {FOREST_GREEN};
        color: white;
        border: none;
        border-radius: 9px;
        font-size: 16px;
        font-weight: 600;
        padding: 11px 20px;
        width: 100%;
    }}

    .stButton > button:hover {{
        background-color: {DARK_GREEN};
        color: white;
    }}

    /* Selectbox */
    div[data-baseweb="select"] > div {{
        border-radius: 8px;
        border-color: {SAGE_GREEN};
    }}

    /* Info */
    .info {{
        background-color: {LIGHT_SAGE};
        border-left: 5px solid {CORAL};
        padding: 15px 18px;
        border-radius: 8px;
        color: #40594A;
        margin-top: 20px;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        color: #708076;
        font-size: 13px;
        padding-top: 30px;
        margin-top: 35px;
        border-top: 1px solid {LIGHT_BORDER};
    }}

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header">
        <h1>📈 Stock Price Prediction</h1>
        <p>
            LSTM-based stock price forecasting using historical market data
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# STOCK MODELS
# ============================================================

stock_models = {

    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "model": "stock_price_lstm.keras",
        "scaler": "stock_scaler.pkl"
    },

    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "model": "tcs_price_lstm.keras",
        "scaler": "tcs_scaler.pkl"
    },

    "INFY.NS": {
        "name": "Infosys",
        "model": "infy_price_lstm.keras",
        "scaler": "infy_scaler.pkl"
    },

    "HDFCBANK.NS": {
        "name": "HDFC Bank",
        "model": "hdfcbank_price_lstm.keras",
        "scaler": "hdfcbank_scaler.pkl"
    },

    "ICICIBANK.NS": {
        "name": "ICICI Bank",
        "model": "icicibank_price_lstm.keras",
        "scaler": "icicibank_scaler.pkl"
    },

    "SBIN.NS": {
        "name": "State Bank of India",
        "model": "sbin_price_lstm.keras",
        "scaler": "sbin_scaler.pkl"
    }
}


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## 📈 Stock Price Prediction")

st.sidebar.markdown("---")

st.sidebar.markdown("### Stock Selection")

ticker = st.sidebar.selectbox(
    "Select Stock",
    list(stock_models.keys())
)

selected_stock = stock_models[ticker]

st.sidebar.markdown("---")

st.sidebar.markdown("### Model Information")

st.sidebar.write("**Algorithm:** LSTM")
st.sidebar.write("**Lookback Period:** 60 days")
st.sidebar.write("**Data Source:** Yahoo Finance")
st.sidebar.write("**Forecast:** Next price")

st.sidebar.markdown("---")

st.sidebar.markdown("### Supported Stocks")

for stock in stock_models:
    st.sidebar.write(f"• {stock}")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Educational machine-learning project."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_prediction_model(model_path, scaler_path):

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    return model, scaler


model, scaler = load_prediction_model(
    selected_stock["model"],
    selected_stock["scaler"]
)


# ============================================================
# SELECTED STOCK
# ============================================================

st.markdown(
    f"""
    <div class="section-title">
        Selected Stock — {selected_stock["name"]}
    </div>
    """,
    unsafe_allow_html=True
)

st.write(f"Ticker: **{ticker}**")


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("Generate Prediction"):

    # --------------------------------------------------------
    # DOWNLOAD DATA
    # --------------------------------------------------------

    with st.spinner("Fetching latest market data..."):

        data = yf.download(
            ticker,
            period="5y",
            auto_adjust=False
        )


    # --------------------------------------------------------
    # CHECK DATA
    # --------------------------------------------------------

    if data.empty:

        st.error(
            f"Unable to retrieve data for {ticker}."
        )

        st.stop()


    # --------------------------------------------------------
    # CLOSE PRICE
    # --------------------------------------------------------

    if isinstance(data.columns, pd.MultiIndex):

        close_data = data["Close"].iloc[:, 0]

    else:

        close_data = data["Close"]


    close_data = close_data.dropna()


    # --------------------------------------------------------
    # CHECK 60 DAYS
    # --------------------------------------------------------

    if len(close_data) < 60:

        st.error(
            "At least 60 trading days are required."
        )

        st.stop()


    # ========================================================
    # CURRENT PRICE
    # ========================================================

    current_price = float(
        close_data.iloc[-1]
    )


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    last_60_days = (
        close_data.values[-60:]
        .reshape(-1, 1)
    )

    scaled_data = scaler.transform(
        last_60_days
    )

    X_input = scaled_data.reshape(
        1,
        60,
        1
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    predicted_scaled = model.predict(
        X_input,
        verbose=0
    )

    predicted_price = float(
        scaler.inverse_transform(
            predicted_scaled
        )[0][0]
    )


    # ========================================================
    # CHANGE
    # ========================================================

    difference = (
        predicted_price -
        current_price
    )

    percentage_change = (
        difference /
        current_price
    ) * 100


    # ========================================================
    # PREDICTION SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Summary</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Current Price",
            f"₹{current_price:,.2f}"
        )


    with col2:

        st.metric(
            "Predicted Next Price",
            f"₹{predicted_price:,.2f}"
        )


    with col3:

        st.metric(
            "Predicted Change",
            f"{percentage_change:.2f}%",
            delta=f"₹{difference:,.2f}"
        )


    # ========================================================
    # HISTORICAL PRICE
    # ========================================================

    st.markdown(
        '<div class="section-title">Historical Price Trend</div>',
        unsafe_allow_html=True
    )

    st.line_chart(
        close_data,
        height=400
    )


    # ========================================================
    # RECENT PRICE & PREDICTION
    # ========================================================

    st.markdown(
        '<div class="section-title">Recent Price & Prediction</div>',
        unsafe_allow_html=True
    )

    recent_prices = close_data.tail(60)


    fig, ax = plt.subplots(
        figsize=(12, 5)
    )


    ax.plot(
        recent_prices.index,
        recent_prices.values,
        linewidth=2,
        label="Historical Price"
    )


    ax.scatter(
        recent_prices.index[-1],
        predicted_price,
        s=90,
        label="Predicted Next Price"
    )


    ax.set_xlabel("Date")

    ax.set_ylabel("Price (₹)")

    ax.set_title(
        f"{selected_stock['name']} - Recent Price"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    st.pyplot(fig)


    # ========================================================
    # MODEL DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">Model Details</div>',
        unsafe_allow_html=True
    )

    detail1, detail2, detail3 = st.columns(3)


    with detail1:

        st.write("**Algorithm**")
        st.write("Long Short-Term Memory (LSTM)")


    with detail2:

        st.write("**Lookback Window**")
        st.write("60 trading days")


    with detail3:

        st.write("**Forecast Type**")
        st.write("One-step prediction")


    # ========================================================
    # DISCLAIMER
    # ========================================================

    st.markdown(
        """
        <div class="info">

        <strong>Important:</strong><br>

        This prediction is generated by a machine-learning
        model trained on historical stock-price data.
        It is intended for educational and demonstration
        purposes and should not be interpreted as a
        guaranteed future price or trading signal.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        Stock Price Prediction &nbsp;•&nbsp;
        LSTM Machine Learning Project<br>
        Python • TensorFlow • Streamlit • Yahoo Finance
    </div>
    """,
    unsafe_allow_html=True
)