import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

def show_visualization():
    st.title("📈 Advanced Dataset Visualization")

    df = pd.read_csv("data/processed/cleaned_data.csv")

    # ================= FILTER =================
    st.sidebar.subheader("🔍 Filters")

    selected_status = st.sidebar.multiselect(
        "Filter by Health Status",
        df["Plant_Health_Status"].unique(),
        default=df["Plant_Health_Status"].unique()
    )

    filtered_df = df[df["Plant_Health_Status"].isin(selected_status)]

    st.success(f"Showing {len(filtered_df)} records")

    st.markdown("---")

    # ================= HISTOGRAM =================
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Health Distribution")
        fig1 = px.histogram(filtered_df, x="Plant_Health_Status", color="Plant_Health_Status")
        fig1.update_layout(template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Feature Distribution")
        feature = st.selectbox("Select Feature", filtered_df.select_dtypes(include=['int64','float64']).columns)
        fig2 = px.histogram(filtered_df, x=feature)
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ================= SCATTER =================
    st.subheader("🔵 Feature Relationship")

    num_cols = filtered_df.select_dtypes(include=['int64','float64']).columns

    x_axis = st.selectbox("X-axis", num_cols)
    y_axis = st.selectbox("Y-axis", num_cols)

    fig3 = px.scatter(filtered_df, x=x_axis, y=y_axis, color="Plant_Health_Status")
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)
