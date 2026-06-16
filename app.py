import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import data_manager, model_engine, xai_engine, llm_service, eda_engine

# --- Page Config & CSS ---
st.set_page_config(page_title="TALENT XAI Platform", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for "Cards" and Clean Design
st.markdown("""
<style>
    .card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border: 1px solid #eee;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
        background-color: #ffffff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Explainable AI for Tabular Data")

# Initialize Session State
if "df" not in st.session_state: st.session_state.df = None
if "X_train" not in st.session_state: st.session_state.X_train = None
if "X_val" not in st.session_state: st.session_state.X_val = None
if "X_test" not in st.session_state: st.session_state.X_test = None
if "y_train" not in st.session_state: st.session_state.y_train = None
if "y_val" not in st.session_state: st.session_state.y_val = None
if "y_test" not in st.session_state: st.session_state.y_test = None
if "info" not in st.session_state: st.session_state.info = None
if "trained_models" not in st.session_state: st.session_state.trained_models = {}
if "leaderboard" not in st.session_state: st.session_state.leaderboard = pd.DataFrame()
if "champion_model" not in st.session_state: st.session_state.champion_model = None
if "transformers" not in st.session_state: st.session_state.transformers = None
if "X_custom_test" not in st.session_state: st.session_state.X_custom_test = None
if "y_custom_test" not in st.session_state: st.session_state.y_custom_test = None

# ==========================================
# SIDEBAR (Control & Data)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/brain.png", width=100)
    st.header("⚙️ Control Panel")
    
    # Section 1: Data Upload
    st.subheader("1. Data Upload")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = data_manager.load_data(uploaded_file)
            st.session_state.df = df
            st.success(f"Data Loaded: {df.shape}")
        except Exception as e:
            st.error(f"Error: {e}")


    # Section 2: Custom Test Set
    if st.session_state.info is not None:
        st.divider()
        st.subheader("2. Custom Test Set")
        custom_test_file = st.file_uploader("Upload Test CSV (Optional)", type=["csv"], key="custom_test")
        
        if custom_test_file is not None:
            try:
                custom_df = data_manager.load_data(custom_test_file)
                if custom_df is not None:
                    # Process using info and transformers
                    X_cust, y_cust = data_manager.process_custom_test_set(
                        custom_df, st.session_state.info, st.session_state.transformers
                    )
                    st.session_state.X_custom_test = X_cust
                    st.session_state.y_custom_test = y_cust
                    st.success(f"Custom Test Set Ready: {X_cust.shape}")
            except Exception as e:
                st.error(f"Processing Error: {e}")

    # Section 3: Instance Selector
    X_to_use = st.session_state.X_test
    y_to_use = st.session_state.y_test
    
    label_prefix = "Val/Test"
    if st.session_state.X_custom_test is not None:
        use_custom = st.checkbox("Use Custom Test Set", value=True)
        if use_custom:
            X_to_use = st.session_state.X_custom_test
            y_to_use = st.session_state.y_custom_test
            label_prefix = "Custom"

    if X_to_use is not None:
        st.divider()
        st.subheader("3. Explain Instance")
        row_idx = st.number_input(
            f"{label_prefix} Instance ID", 
            min_value=0, 
            max_value=len(X_to_use)-1, 
            value=0
        )


# ==========================================
# HELPERS
# ==========================================
def show_local_prediction(model, instance, y_true, info):
    """Reusable component to display instance features, prediction, and truth."""
    # Determine task type
    task_type = info['task_type']
    target_mapping = info.get('target_mapping')
    
    # 1. Prediction logic
    pred = model_engine.predict(model, instance)[0]
    prob = model_engine.predict_proba(model, instance)
    if prob is not None:
        prob = prob[0]
            
    # 2. Display Features (in an expander to save space)
    with st.expander("📝 Selected Instance Features", expanded=False):
        st.dataframe(instance)
        
    # 3. Display Prediction & True Label
    st.markdown("#### 🎯 Prediction vs Truth")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("**Predicted**")
        label = target_mapping.get(str(pred), pred) if target_mapping else pred
        st.metric("Class", str(label))
            
    with c2:
        st.markdown("**True Label**")
        label = target_mapping.get(str(y_true), y_true) if target_mapping else y_true
        st.metric("Class", str(label))
            
    with c3:
        st.markdown("**Status**")
        match = "✅ Match" if pred == y_true else "❌ Mismatch"
        st.write(f"### {match}")
        if prob is not None:
            st.write(f"Confidence: {np.max(prob):.2%}")

# ==========================================
# MAIN AREA (Workflow)
# ==========================================

# TABS for workflow
tab_pre, tab_intrinsic, tab_post = st.tabs(["📊 Pre-modelling Explainability", "🔍 Intrinsic Interpretation (Glass Box)", "🧠 Post-hoc Explainability (Black Box)"])

# --- TAB 1: Pre-modelling ---
with tab_pre:
    if st.session_state.df is not None:
        col_set, col_eda = st.columns([1, 2])
        
        with col_set:
            st.markdown('<div class="card"><h4>⚙️ Processing Pipeline</h4></div>', unsafe_allow_html=True)
            target_col = st.selectbox("🎯 Target Variable", st.session_state.df.columns)
            
            with st.expander("🛠️ Advanced Settings", expanded=True):
                impute_opt = st.selectbox("Imputation", ["Mean", "Median"])
                scaling_opt = st.selectbox("Scaling", ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
                encoding_opt = st.selectbox("Encoding", ["Label Encoding", "One-Hot Encoding"])
                
                st.divider()
                st.markdown("##### Feature Engineering")
                drop_dups = st.checkbox("Remove Duplicates", value=True)
                
                # Feature Engineering section remains for Remove Duplicates

                st.divider()
                st.markdown("##### Imbalance")
                imb_opt = st.selectbox("Class Balancing", ["None", "Undersampling", "Oversampling", "SMOTE"])
            
            if st.button("🚀 Process & Partition Data", use_container_width=True, type="primary"):
                with st.spinner("Processing..."):
                    try:
                        X_tr, X_val, X_te, y_tr, y_val, y_te, info, trans = data_manager.preprocess_data(
                            st.session_state.df, target_col,
                            impute_strategy=impute_opt,
                            scaling_strategy=scaling_opt,
                            encoding_strategy=encoding_opt,
                            handle_imbalance=imb_opt if imb_opt != "None" else None,
                            drop_duplicates=drop_dups,
                            random_state=42
                        )
                        st.session_state.X_train, st.session_state.X_val, st.session_state.X_test = X_tr, X_val, X_te
                        st.session_state.y_train, st.session_state.y_val, st.session_state.y_test = y_tr, y_val, y_te
                        st.session_state.info = info
                        st.session_state.transformers = trans
                        st.session_state.X_custom_test = None # Reset custom test on new training
                        st.success("Data Ready!")
                        
                        # Display Leaks if any
                        if info.get('potential_leaks'):
                            st.warning("⚠️ Potential Target Leakage Detected!")
                            for col, reason in info['potential_leaks'].items():
                                st.write(f"- **{col}**: {reason}")
                            st.info("Consider dropping these columns or refining your dataset to prevent model 'cheating'.")
                        
                        # Display SMOTE Comparison if applicable
                        if info.get('smote_stats'):
                            st.markdown("---")
                            st.markdown("#### ⚖️ Rebalancing Validation")
                            fig_smote = data_manager.plot_smote_comparison(info['smote_stats'])
                            if fig_smote:
                                st.plotly_chart(fig_smote, use_container_width=True)
                            
                    except Exception as e:
                        st.error(str(e))
            
            # Removed legacy pipeline stages display

            if st.session_state.X_test is not None:
                st.divider()
                st.markdown("#### 📦 Export Test Set")
                # Export raw data (unprocessed) for easier re-upload
                target_col = st.session_state.info['target_col']
                # Get the indices used for the test set
                test_indices = st.session_state.X_test.index
                test_df_raw = st.session_state.df.loc[test_indices].copy()
                
                csv = test_df_raw.to_csv(index=False).encode('utf-8')
                st.download_button("Download test_dataset.csv", csv, "test_dataset.csv", "text/csv", use_container_width=True)

        with col_eda:
            eda_engine.show_eda_page(st.session_state.df)
    else:
        st.info("Please upload a dataset in the sidebar to begin.")

# --- TAB 2: Intrinsic Interpretation ---
with tab_intrinsic:
    st.markdown("#### Intrinsic Interpretation (Glass Box Models)")
    
    if st.session_state.info:
        col_glass_sel, col_glass_res = st.columns([1, 2])
        
        with col_glass_sel:
            st.markdown('<div class="card"><h5>Models</h5></div>', unsafe_allow_html=True)
            glass_options = ['Logistic Regression', 'Decision Tree', 'KNN']
            if st.session_state.info['task_type'] == 'regression':
                glass_options = ['Linear Regression', 'Decision Tree', 'KNN']
            selected_glass = st.multiselect("Select Models", glass_options, default=glass_options[:2])
            
            if st.button("🚀 Train Glass Box Models", type="primary"):
                results = []
                with st.spinner("Training..."):
                    for model_name in selected_glass:
                        try:
                            # Train
                            model = model_engine.train_model(
                                st.session_state.X_train, st.session_state.y_train, 
                                st.session_state.X_val, st.session_state.y_val, 
                                model_name, st.session_state.info['task_type']
                            )
                            # Save to a specific dict for Glass Box to avoid overwrite/confusion
                            if "glass_models" not in st.session_state: st.session_state.glass_models = {}
                            st.session_state.glass_models[model_name] = model
                            
                            # Evaluate on Validation
                            m_val = model_engine.evaluate_model(
                                model, st.session_state.X_val, st.session_state.y_val, st.session_state.info['task_type']
                            )
                            # Evaluate on Test
                            m_test = model_engine.evaluate_model(
                                model, st.session_state.X_test, st.session_state.y_test, st.session_state.info['task_type']
                            )
                            
                            # Merge metrics
                            combined = {'Model': model_name}
                            for k in ['Accuracy', 'F1 Score', 'Recall', 'ROC AUC', 'R2', 'RMSE']:
                                if k in m_val: combined[f'{k} (Val)'] = m_val[k]
                                if k in m_test: combined[f'{k} (Test)'] = m_test[k]
                            
                            combined['Confusion Matrix'] = m_test.get('Confusion Matrix') # Only Test
                            results.append(combined)
                        except Exception as e:
                            st.error(f"Error {model_name}: {e}")
                
                if results:
                    st.session_state.glass_leaderboard = pd.DataFrame(results).set_index("Model")
                    st.success("Training Complete!")

        with col_glass_res:
             if "glass_leaderboard" in st.session_state and not st.session_state.glass_leaderboard.empty:
                st.markdown("##### Performance Metrics")
                # Highlight columns (Test columns)
                test_cols = [c for c in st.session_state.glass_leaderboard.columns if '(Test)' in c]
                st.dataframe(st.session_state.glass_leaderboard.style.highlight_max(axis=0, subset=test_cols, color='lightgreen'))
                
                # --- Confusion Matrix for Glass Box ---
                if st.session_state.info['task_type'] != 'regression':
                    st.divider()
                    st.markdown("##### 📉 Confusion Matrices (Test Set)")
                    cm_cols = st.columns(len(st.session_state.glass_leaderboard))
                    for i, (m_name, row) in enumerate(st.session_state.glass_leaderboard.iterrows()):
                        with cm_cols[i % len(cm_cols)]:
                            st.write(f"**{m_name}**")
                            cm = row.get('Confusion Matrix')
                            if cm:
                                # Convert list to array for display
                                cm_arr = np.array(cm)
                                fig_cm = px.imshow(cm_arr, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                                  x=[str(c) for c in st.session_state.info['target_mapping'].values()],
                                                  y=[str(c) for c in st.session_state.info['target_mapping'].values()],
                                                  color_continuous_scale='Blues')
                                fig_cm.update_layout(width=250, height=250, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                                fig_cm.update_coloraxes(showscale=False)
                                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.divider()
                st.markdown("##### Interpretation")
                # Select model to interpret
                model_to_interpret = st.selectbox("Select Model to Interpret", st.session_state.glass_leaderboard.index)
                chosen_model = st.session_state.glass_models[model_to_interpret]
                
                # Instance prediction display
                if X_to_use is not None:
                    st.divider()
                    instance = X_to_use.iloc[[row_idx]]
                    y_true = y_to_use.iloc[row_idx] if y_to_use is not None else -1
                    show_local_prediction(chosen_model, instance, y_true, st.session_state.info)
                
                st.divider()
                if "Regression" in model_to_interpret:
                    st.write("**Log-Odds / Coefficients**")
                    df_coeffs = model_engine.get_lr_coeffs(chosen_model, st.session_state.info['feature_names'])
                    if df_coeffs is not None:
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(data=df_coeffs.head(10), x='Coefficient', y='Feature', ax=ax, palette='viridis')
                        st.pyplot(fig)
                    else:
                        st.info("Coefficients not available for this model type.")
                        
                elif "Decision Tree" in model_to_interpret:
                    st.write("**Tree Structure (Text)**")
                    text_rep = model_engine.get_dt_text(chosen_model, st.session_state.info['feature_names'])
                    st.text(text_rep)
                    
                elif "KNN" in model_to_interpret:
                    st.write("**Nearest Neighbors**")
                    if st.session_state.X_test is not None:
                        instance = st.session_state.X_test.iloc[[row_idx]] # Use shared row_idx
                        dist, ind = model_engine.get_knn_neighbors(chosen_model, instance, st.session_state.X_train)
                        
                        st.write(f"Neighbors for Instance {row_idx}:")
                        # Show neighbors from X_train (need to map back to df if possible, or just show values)
                        # We have X_train as dataframe? No, X_train is numpy in data_manager if split? 
                        # Wait, data_manager returns pandas df for X_train.
                        
                        neighbors_df = st.session_state.X_train.iloc[ind].copy()
                        neighbors_df['Distance'] = dist
                        st.dataframe(neighbors_df)
                    else:
                        st.warning("No test data available for instance selection.")

    else:
        st.info("Please upload and process data first.")

# --- TAB 3: Post-hoc Explainability ---
with tab_post:
    st.markdown("#### Post-hoc Explainability (TALENT / Black Box)")
    
    if st.session_state.info:
        col_black_sel, col_black_res = st.columns([1, 2]) # Reuse layout concept
        
        with col_black_sel:
            st.markdown('<div class="card"><h5>TALENT Models</h5></div>', unsafe_allow_html=True)
            black_options = ['FT-Transformer', 'ResNet', 'SwitchTab', 'TabNet', 'TabR', 'TabPFN', 'MLP']
            selected_black = st.multiselect("Select Deep Models", black_options, default=['FT-Transformer', 'ResNet'])
            
            if st.button("🚀 Train & Benchmark", type="primary"):
                results = []
                progress = st.progress(0)
                
                # Reset previous state to avoid stale data
                st.session_state.trained_models = {}
                st.session_state.leaderboard = pd.DataFrame()
                st.session_state.global_imp = None
                st.session_state.local_exp = None

                for i, model_name in enumerate(selected_black):
                    try:
                        model = model_engine.train_model(
                            st.session_state.X_train, st.session_state.y_train, 
                            st.session_state.X_val, st.session_state.y_val, 
                            model_name, st.session_state.info['task_type']
                        )
                        st.session_state.trained_models[model_name] = model
                        
                        # Evaluate on Validation
                        m_val = model_engine.evaluate_model(
                            model, st.session_state.X_val, st.session_state.y_val, st.session_state.info['task_type']
                        )
                        # Evaluate on Test
                        m_test = model_engine.evaluate_model(
                            model, st.session_state.X_test, st.session_state.y_test, st.session_state.info['task_type']
                        )
                        
                        # Merge metrics
                        combined = {'Model': model_name}
                        for k in ['Accuracy', 'F1 Score', 'Recall', 'ROC AUC', 'R2', 'RMSE']:
                            if k in m_val: combined[f'{k} (Val)'] = m_val[k]
                            if k in m_test: combined[f'{k} (Test)'] = m_test[k]
                        
                        combined['Confusion Matrix'] = m_test.get('Confusion Matrix') # Only Test
                        results.append(combined)
                    except Exception as e:
                        st.error(f"Error {model_name}: {e}")
                    progress.progress((i+1)/len(selected_black))
                
                if results:
                    st.session_state.leaderboard = pd.DataFrame(results).set_index("Model")
                    st.success("Benchmarking Complete!")

        # --- PERSISTENT RESULTS DISPLAY ---
        with col_black_res:
             if not st.session_state.leaderboard.empty:
                st.markdown("##### Leaderboard")
                test_cols = [c for c in st.session_state.leaderboard.columns if '(Test)' in c]
                st.dataframe(st.session_state.leaderboard.style.highlight_max(axis=0, subset=test_cols, color='lightgreen'))
                
                # --- Confusion Matrix for Black Box ---
                if st.session_state.info['task_type'] != 'regression':
                    st.divider()
                    st.markdown("##### 📉 Confusion Matrices (Test Set)")
                    cm_cols = st.columns(min(3, len(st.session_state.leaderboard)))
                    for i, (m_name, row) in enumerate(st.session_state.leaderboard.iterrows()):
                        with cm_cols[i % 3]:
                            st.write(f"**{m_name}**")
                            cm = row.get('Confusion Matrix')
                            if cm:
                                cm_arr = np.array(cm)
                                fig_cm = px.imshow(cm_arr, text_auto=True, labels=dict(x="Predicted", y="Actual"),
                                                  x=[str(c) for c in st.session_state.info['target_mapping'].values()],
                                                  y=[str(c) for c in st.session_state.info['target_mapping'].values()],
                                                  color_continuous_scale='Greens')
                                fig_cm.update_layout(width=250, height=250, margin=dict(l=10, r=10, t=30, b=10), showlegend=False)
                                fig_cm.update_coloraxes(showscale=False)
                                st.plotly_chart(fig_cm, use_container_width=True)
                
                st.divider()
                st.markdown("##### Explainability Suite")
                champ = st.selectbox("Select Model to Explain", st.session_state.leaderboard.index)
                st.session_state.champion_model = st.session_state.trained_models[champ]
                
                # Explainability Tabs
                tab_global, tab_local = st.tabs(["🌍 Global Explainability", "📍 Local Explainability"])
                    
                # --- Global: Permutation Importance ---
                with tab_global:
                    st.markdown("##### 📉 Permutation Importance")
                    scoring_method = st.selectbox("Scoring Metric", ["accuracy", "f1", "roc_auc"] if st.session_state.info['task_type'] == 'classification' else ["neg_root_mean_squared_error", "r2"])
                    
                    if st.button("Calculate Importance", use_container_width=True):
                        with st.spinner("Calculating..."):
                            df_imp = xai_engine.compute_permutation_importance(
                                st.session_state.champion_model,
                                st.session_state.X_val,
                                st.session_state.y_val,
                                st.session_state.info['task_type'],
                                scoring=scoring_method
                            )
                            if not df_imp.empty:
                                st.session_state.global_imp = df_imp
                            else:
                                st.error("Failed to calculate importance.")
                    
                    if st.session_state.get("global_imp") is not None:
                        df_plot = st.session_state.global_imp.sort_values(by='Importance', ascending=True)
                        fig = px.bar(
                            df_plot, 
                            x='Importance', 
                            y='Feature', 
                            orientation='h',
                            title=f"Global Importance ({scoring_method})",
                            color='Importance',
                            color_continuous_scale='Viridis',
                            error_x='Std'
                        )
                        fig.update_layout(height=600, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)

                # --- Local: SHAP & LIME ---
                with tab_local:
                    if X_to_use is not None:
                        instance = X_to_use.iloc[[row_idx]]
                        y_true = y_to_use.iloc[row_idx] if y_to_use is not None else -1
                        
                        st.markdown(f'<div class="card"><strong>Explaining {label_prefix} ID: {row_idx}</strong></div>', unsafe_allow_html=True)
                        
                        # Show prediction display here too
                        show_local_prediction(st.session_state.champion_model, instance, y_true, st.session_state.info)
                        
                        st.divider()
                        st.markdown("#### 🛠️ Explanation Tuning & Generation")
                        c1, c2, c3 = st.columns(3)
                        with c1: l_samples = st.number_input("LIME Samples", 500, 10000, 5000, 500)
                        with c2: l_depth = st.number_input("Max Features", 3, 20, 10)
                        with c3: l_metric = st.selectbox("Distance Metric", ["euclidean", "cosine", "l1"])
                        
                        if st.button("🔍 Generate Local Explanations", use_container_width=True, type="primary"):
                            with st.spinner("Computing..."):
                                try:
                                    # SHAP
                                    shap_vals, expected_val = xai_engine.compute_shap(
                                        st.session_state.champion_model,
                                        st.session_state.X_train,
                                        instance,
                                        st.session_state.info['task_type']
                                    )
                                    # LIME
                                    lime_exp = xai_engine.compute_lime(
                                        st.session_state.champion_model,
                                        st.session_state.X_train,
                                        instance,
                                        st.session_state.info['feature_names'],
                                        st.session_state.info['task_type'],
                                        num_samples=l_samples,
                                        num_features=l_depth,
                                        distance_metric=l_metric
                                    )
                                    st.session_state.local_exp = {
                                        "shap_vals": shap_vals,
                                        "expected_val": expected_val,
                                        "lime_exp": lime_exp,
                                        "instance": instance,
                                        "idx": row_idx
                                    }
                                except Exception as e:
                                    st.error(f"XAI Error: {e}")
                                    st.exception(e)

                        if st.session_state.get("local_exp") is not None and st.session_state.local_exp["idx"] == row_idx:
                            exp = st.session_state.local_exp
                            
                            col_shap, col_lime = st.columns(2)
                            
                            with col_shap:
                                st.markdown("##### 📊 SHAP waterfall")
                                try:
                                    # Create SHAP Explanation object for waterfall plot
                                    sv = exp["shap_vals"]
                                    # sv is usually (1, n_features) or (n_features,)
                                    if sv.ndim > 1: sv = sv[0]
                                    
                                    exp_obj = shap.Explanation(
                                        values=sv,
                                        base_values=exp["expected_val"],
                                        data=exp["instance"].values[0],
                                        feature_names=st.session_state.info['feature_names']
                                    )
                                    
                                    fig_wf, ax_wf = plt.subplots(figsize=(8, 6))
                                    shap.plots.waterfall(exp_obj, show=False)
                                    st.pyplot(fig_wf)
                                except Exception as e:
                                    st.error(f"Waterfall error: {e}")

                            with col_lime:
                                st.markdown("##### 🎯 LIME Explanation")
                                # Convert LIME to Plotly
                                lime_data = exp["lime_exp"].as_list()
                                df_lime = pd.DataFrame(lime_data, columns=['Feature Condition', 'Effect'])
                                df_lime['Color'] = df_lime['Effect'].apply(lambda x: 'Positive Contribution' if x > 0 else 'Negative Contribution')
                                
                                fig_lime = px.bar(
                                    df_lime, 
                                    x='Effect', 
                                    y='Feature Condition', 
                                    orientation='h',
                                    color='Color',
                                    color_discrete_map={'Positive Contribution': '#2ecc71', 'Negative Contribution': '#e74c3c'},
                                    title="Local Feature Impact (LIME)"
                                )
                                fig_lime.update_layout(showlegend=False, height=450, margin=dict(l=20, r=20, t=40, b=20))
                                st.plotly_chart(fig_lime, use_container_width=True)

                            with st.expander("🌊 SHAP Force Plot"):
                                try:
                                    # Force plot usually works better with HTML in Streamlit
                                    # But let's try matplotlib first for safety/simplicity in this environment
                                    fig_force, ax_force = plt.subplots(figsize=(10, 3))
                                    shap.plots.force(
                                        exp["expected_val"], 
                                        sv, 
                                        feature_names=st.session_state.info['feature_names'],
                                        matplotlib=True, 
                                        show=False
                                    )
                                    st.pyplot(plt.gcf())
                                except Exception as e:
                                    st.info(f"Force plot display error: {e}")
                    else:
                        st.warning("No Test Data.")
    else:
        st.info("Process Data First.")
