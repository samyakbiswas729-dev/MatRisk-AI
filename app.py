import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# LOGIN SYSTEM
# -------------------------
users = {
    "admin": {"password": "admin123", "role": "Admin"},
    "analyst": {"password": "analyst123", "role": "Analyst"},
    "viewer": {"password": "viewer123", "role": "Viewer"}
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login():
    st.title("🔐 MatRisk AI Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u]["password"] == p:
            st.session_state.logged_in = True
            st.session_state.role = users[u]["role"]
            st.session_state.username = u
            st.rerun()
        else:
            st.error("Invalid credentials")

def logout():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

# -------------------------
# MAIN APP
# -------------------------
if not st.session_state.logged_in:
    login()

else:
    logout()

    st.set_page_config(layout="wide")
    st.title("🚀 MatRisk AI - Stable Time Intelligence Dashboard")

    df = pd.read_csv("data.csv")

    # Add variation
    np.random.seed(42)
    df["price"] = df["price"] + np.random.randn(len(df)) * 3

    # -------------------------
    # SIDEBAR
    # -------------------------
    commodity = st.sidebar.selectbox(
        "Commodity",
        sorted(df["commodity"].unique())
    )

    min_time = int(df["time"].min())
    max_time = int(df["time"].max())

    time_range = st.sidebar.slider(
        "Select Time Range",
        min_time,
        max_time,
        (min_time, max_time)
    )

    df = df[
        (df["commodity"] == commodity) &
        (df["time"] >= time_range[0]) &
        (df["time"] <= time_range[1])
    ].copy()

    df = df.sort_values("time")

    # -------------------------
    # HANDLE SMALL DATA
    # -------------------------
    if len(df) < 2:
        st.warning("⚠️ Not enough data points for analysis. Please expand time range.")
        st.dataframe(df)
        st.stop()

    # -------------------------
    # FEATURES
    # -------------------------
    df["MQI"] = 0.6 * df["density"] + 0.4 * df["elasticity"]
    df["returns"] = df["price"].pct_change()
    df["moving_avg"] = df["price"].rolling(2).mean()
    df["volatility"] = df["price"].rolling(2).std()

    df.bfill(inplace=True)

    # -------------------------
    # MODEL
    # -------------------------
    X = df[["MQI", "moving_avg"]]
    y = df["price"]

    model = RandomForestRegressor()
    model.fit(X, y)
    pred = model.predict(X)

    # -------------------------
    # KPI
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Price", round(df["price"].mean(),2))
    col2.metric("MQI", round(df["MQI"].mean(),2))
    col3.metric("Volatility", round(df["volatility"].mean(),2))
    col4.metric("Returns %", round(df["returns"].mean()*100,2))

    # -------------------------
    # TABS
    # -------------------------
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Overview","Analytics","Risk","AI"]
    )

    # =========================
    # OVERVIEW
    # =========================
    with tab1:
        st.subheader("📊 Time-Based Insights")

        slope = np.polyfit(df["time"], df["price"], 1)[0]

        if slope > 0:
            st.success("📈 Uptrend")
        else:
            st.warning("📉 Downtrend")

        momentum = df["price"].iloc[-1] - df["price"].iloc[0]
        st.write("Momentum:", round(momentum,2))

    # =========================
    # ANALYTICS
    # =========================
    with tab2:
        st.subheader("Price vs Moving Avg")

        fig, ax = plt.subplots()
        ax.plot(df["time"], df["price"], label="Price")
        ax.plot(df["time"], df["moving_avg"], label="Moving Avg")
        ax.legend()
        st.pyplot(fig)

        st.subheader("Returns")

        fig2, ax2 = plt.subplots()
        ax2.plot(df["time"], df["returns"])
        st.pyplot(fig2)

    # =========================
    # RISK
    # =========================
    with tab3:
        st.subheader("Risk Analysis")

        vol = df["volatility"].iloc[-1]

        if vol > 5:
            st.error("High Risk")
        elif vol > 2:
            st.warning("Moderate Risk")
        else:
            st.success("Low Risk")

    # =========================
    # AI
    # =========================
    with tab4:
        if st.session_state.role == "Admin":

            st.subheader("Prediction")

            fig3, ax3 = plt.subplots()
            ax3.plot(df["time"], y, label="Actual")
            ax3.plot(df["time"], pred, label="Predicted")
            ax3.legend()
            st.pyplot(fig3)

            st.subheader("Forecast")

            if len(pred) > 1:
                if pred[-1] > pred[-2]:
                    st.success("📈 Future: UP")
                else:
                    st.warning("📉 Future: DOWN")
            else:
                st.info("Not enough data for forecast")

        else:
            st.warning("Admin only")

    st.markdown("---")
    st.caption("MatRisk AI | Stable & Robust System")