import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(
    page_title="Enterprise Retail Analytics",
    page_icon="📊",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================
@st.cache_data
def load_data():
    df = pd.read_csv("superstore/superstore.csv", encoding="latin1")
    df = df.drop(columns=["è®°å½æ°"], errors="ignore")
    df["Order.Date"] = pd.to_datetime(df["Order.Date"])
    df["Ship.Date"] = pd.to_datetime(df["Ship.Date"])
    return df

df = load_data()

model = joblib.load("sales_prediction_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# =====================================
# SIDEBAR
# =====================================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dashboard",
        "Sales Prediction",
        "Demand Forecasting",
        "Business Insights"
    ]
)

# =====================================
# DASHBOARD
# =====================================
if page == "Dashboard":

    st.title("🏪 Enterprise Retail Analytics Dashboard")

    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order.ID"].nunique()
    avg_sales = df["Sales"].mean()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("💰 Total Sales", f"${total_sales:,.0f}")
    c2.metric("📈 Total Profit", f"${total_profit:,.0f}")
    c3.metric("📦 Orders", total_orders)
    c4.metric("💵 Avg Sales", f"${avg_sales:.2f}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Sales by Category")

        fig, ax = plt.subplots(figsize=(6,4))

        df.groupby("Category")["Sales"].sum().plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

    with col2:

        st.subheader("Profit by Category")

        fig, ax = plt.subplots(figsize=(6,4))

        df.groupby("Category")["Profit"].sum().plot(
            kind="bar",
            ax=ax
        )

        st.pyplot(fig)

    st.divider()

    st.subheader("Sales by Market")

    fig, ax = plt.subplots(figsize=(8,4))

    df.groupby("Market2")["Sales"].sum().plot(
        kind="bar",
        ax=ax
    )

    st.pyplot(fig)

    st.divider()

    st.subheader("Top 10 Cities by Sales")

    fig, ax = plt.subplots(figsize=(10,4))

    df.groupby("City")["Sales"].sum()\
        .sort_values(ascending=False)\
        .head(10)\
        .plot(kind="bar", ax=ax)

    st.pyplot(fig)

    st.divider()

    st.subheader("Monthly Sales Trend")

    monthly = df.groupby(
        df["Order.Date"].dt.to_period("M")
    )["Sales"].sum()

    fig, ax = plt.subplots(figsize=(12,5))

    monthly.plot(marker="o", ax=ax)

    plt.xticks(rotation=45)

    st.pyplot(fig)
    # =====================================
# SALES PREDICTION
# =====================================
elif page == "Sales Prediction":

    st.title("🤖 Sales Prediction")

    category = st.selectbox(
        "Category",
        encoders["Category"].classes_
    )

    subcategory = st.selectbox(
        "Sub Category",
        encoders["Sub.Category"].classes_
    )

    quantity = st.number_input(
        "Quantity",
        min_value=1,
        max_value=100,
        value=5
    )

    discount = st.slider(
        "Discount",
        0.0,
        0.85,
        0.10
    )

    shipping = st.number_input(
        "Shipping Cost",
        min_value=0.0,
        value=10.0
    )

    market = st.selectbox(
        "Market",
        encoders["Market2"].classes_
    )

    segment = st.selectbox(
        "Segment",
        encoders["Segment"].classes_
    )

    if st.button("Predict Sales"):

        input_df = pd.DataFrame({
            "Category":[encoders["Category"].transform([category])[0]],
            "Sub.Category":[encoders["Sub.Category"].transform([subcategory])[0]],
            "Quantity":[quantity],
            "Discount":[discount],
            "Shipping.Cost":[shipping],
            "Market2":[encoders["Market2"].transform([market])[0]],
            "Segment":[encoders["Segment"].transform([segment])[0]]
        })

        prediction = model.predict(input_df)

        st.success(f"Predicted Sales : ${prediction[0]:.2f}")

# =====================================
# DEMAND FORECASTING
# =====================================
elif page == "Demand Forecasting":

    st.title("📈 Demand Forecasting")

    monthly_sales = df.groupby(
        df["Order.Date"].dt.to_period("M")
    )["Sales"].sum().reset_index()

    monthly_sales["Month"] = range(1, len(monthly_sales)+1)

    X = monthly_sales[["Month"]]
    y = monthly_sales["Sales"]

    lr = LinearRegression()
    lr.fit(X, y)

    future = pd.DataFrame({
        "Month": range(len(monthly_sales)+1, len(monthly_sales)+7)
    })

    future["Predicted Sales"] = lr.predict(future)

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        monthly_sales["Month"],
        y,
        marker="o",
        label="Historical Sales"
    )

    ax.plot(
        future["Month"],
        future["Predicted Sales"],
        marker="o",
        linestyle="--",
        label="Forecast"
    )

    ax.set_xlabel("Month")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Next 6 Months Forecast")

    st.dataframe(future)

# =====================================
# BUSINESS INSIGHTS
# =====================================
elif page == "Business Insights":

    st.title("📊 Business Insights")

    highest_category = df.groupby("Category")["Sales"].sum().idxmax()
    highest_market = df.groupby("Market2")["Sales"].sum().idxmax()
    highest_city = df.groupby("City")["Sales"].sum().idxmax()

    st.success(f"🏆 Highest Sales Category : {highest_category}")
    st.success(f"🌍 Best Market : {highest_market}")
    st.success(f"🏙️ Best City : {highest_city}")

    st.divider()

    st.subheader("Top 10 Products")

    top_products = (
        df.groupby("Product.Name")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.dataframe(top_products)

    st.divider()

    st.subheader("Sales by Segment")

    fig, ax = plt.subplots(figsize=(7,4))

    df.groupby("Segment")["Sales"].sum().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax
    )

    ax.set_ylabel("")

    st.pyplot(fig)

    st.divider()

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Download Dataset",
        csv,
        "superstore.csv",
        "text/csv"
    )