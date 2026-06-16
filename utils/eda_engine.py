import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

def show_eda_page(df=None):
    """
    Function to display the Pre-modelling Explainability page with a unified EDA view.
    """
    if df is None:
        st.info("Please upload a dataset in the sidebar to view analysis.")
        return

    # Dynamic Cardinality Heuristic
    # Re-classify numerical variables with nunique <= 2 as "Categorical/Binary"
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    reclassified_as_cat = []
    for col in numerical_cols:
        if df[col].nunique() <= 2:
            reclassified_as_cat.append(col)
    
    # Update lists for visualization purposes
    viz_categorical = categorical_cols + reclassified_as_cat
    viz_numerical = [c for c in numerical_cols if c not in reclassified_as_cat]

    # 1. Dataset Overview
    st.subheader("📋 Dataset Overview & Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    with st.expander("Show Raw Data (.head)", expanded=False):
        st.dataframe(df.head())

    with st.expander("Data Info (.info)", expanded=False):
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())

    with st.expander("Descriptive Statistics (.describe)", expanded=False):
        st.dataframe(df.describe(include='all'))

    st.divider()

    # 2. Target Analysis
    st.subheader("🎯 Target Variable Analysis")
    target_col = st.selectbox("Select Target Variable", df.columns, key="eda_target_selection")
    
    if target_col:
        col_donut, col_alert = st.columns([2, 1])
        
        target_counts = df[target_col].value_counts()
        total_samples = len(df)
        
        with col_donut:
            fig_target = px.pie(
                values=target_counts.values, 
                names=target_counts.index, 
                hole=0.4,
                title=f"Distribution of {target_col}",
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            st.plotly_chart(fig_target, use_container_width=True)
            
        with col_alert:
            st.markdown("#### Imbalance Status")
            imbalance_detected = False
            for cls, count in target_counts.items():
                ratio = (count / total_samples) * 100
                st.write(f"**Class {cls}**: {ratio:.1f}%")
                if ratio < 20:
                    imbalance_detected = True
            
            if imbalance_detected:
                st.warning("⚠️ **Class Imbalance Detected!** One or more classes represent less than 20% of the total samples. Consider using SMOTE during preprocessing.")
            else:
                st.success("✅ Class distribution is relatively balanced.")

    st.divider()

    # 3. Adaptive Statistical Visualizations
    st.subheader("📈 Statistical Visualizations")
    
    # Univariate Analysis
    st.markdown("#### Univariate Analysis")
    viz_col = st.selectbox("Select Feature for Univariate Analysis", df.columns, key="univariate_feature")
    
    if viz_col:
        if viz_col in viz_numerical:
            # High Cardinality Numerical: Histogram + KDE and Boxplot
            st.markdown(f"**Analyzing Continuous Feature: {viz_col}**")
            
            fig_hist = px.histogram(
                df, x=viz_col, marginal="box", 
                title=f"Distribution of {viz_col}",
                color_discrete_sequence=['skyblue']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        else:
            # Categorical or Low Cardinality Numerical
            st.markdown(f"**Analyzing Categorical/Binary Feature: {viz_col}**")
            fig_count = px.histogram(
                df, x=viz_col, 
                title=f"Count of {viz_col}",
                color=viz_col if df[viz_col].nunique() < 10 else None,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_count, use_container_width=True)

    # Bivariate Analysis
    st.markdown("#### Bivariate Analysis")
    if target_col:
        feature_col = st.selectbox("Select Feature vs Target", [c for c in df.columns if c != target_col])
        
        if feature_col:
            if feature_col in viz_numerical:
                # Continuous vs Target
                st.markdown(f"**Boxplot/Violin: {feature_col} by {target_col}**")
                fig_bi = px.box(
                    df, x=target_col, y=feature_col, color=target_col,
                    title=f"{feature_col} Distribution by {target_col}",
                    points="all"
                )
                st.plotly_chart(fig_bi, use_container_width=True)
            else:
                # Categorical vs Target: Stacked Bar Chart
                st.markdown(f"**Stacked Bar Chart: {feature_col} vs {target_col}**")
                # Group data for stacked bar
                counts = df.groupby([feature_col, target_col]).size().reset_index(name='count')
                counts['percentage'] = counts.groupby(feature_col)['count'].transform(lambda x: (x / x.sum()) * 100)
                
                fig_stacked = px.bar(
                    counts, x=feature_col, y='count', color=target_col,
                    title=f"Distribution of {target_col} across {feature_col} modalities",
                    barmode="stack",
                    text=counts['percentage'].apply(lambda x: f'{x:.1f}%')
                )
                st.plotly_chart(fig_stacked, use_container_width=True)

    # Multivariate Analysis
    st.divider()
    st.subheader("📊 Global Correlation Matrix")
    
    # Process all features for global correlation
    # Copy DF and label encode categorical columns
    corr_df = df.copy()
    label_encoders = {}
    
    # Categorical columns (including those reclassified earlier)
    all_cat = viz_categorical
    
    for col in all_cat:
        le = LabelEncoder()
        # Handle NA for encoding
        temp_col = corr_df[col].astype(str).fillna("Missing")
        corr_df[col] = le.fit_transform(temp_col)
        label_encoders[col] = le

    method = st.selectbox("Correlation Coefficient", ["pearson", "spearman"])
    corr = corr_df.corr(method=method)
    
    fig_corr = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        title=f"Global Correlation Matrix ({method.capitalize()})",
        color_continuous_scale='RdBu_r',
        range_color=[-1, 1]
    )
    st.plotly_chart(fig_corr, use_container_width=True)
