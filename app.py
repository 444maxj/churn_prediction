# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='45' fill='%231d4ed8'/></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Internal imports
from src.prediction import predict_churn, get_retention_recommendations
from src.ui_components import (
    apply_custom_css,
    render_header,
    render_client_form,
    render_prediction_result,
    render_gauge_chart,
    render_confusion_matrix_plotly,
    render_roc_plotly,
    render_feature_importance_plotly,
    render_recommendations,
)
from src.dashboard import (
    render_kpi_row,
    render_churn_overview,
    render_financial_analysis,
    render_demographic_analysis,
    render_service_analysis,
    render_funnel_analysis,
)

apply_custom_css()

# ─────────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_all_models(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    
    models = {}
    try:
        models['XGBoost'] = joblib.load('models/xgb_model.pkl')
        models['CatBoost'] = joblib.load('models/cat_model.pkl')
    except FileNotFoundError:
        st.warning("Pre-trained XGBoost or CatBoost models not found. Running baseline training for fallback.")
        
    y_tr = np.array(y_train).astype(int).ravel()
    
    # Baseline models for comparison
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_tr)
    models['Logistic Regression'] = lr
    
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_tr)
    models['Random Forest'] = rf
    
    return models

# ─────────────────────────────────────────────────────────────
# DATA LOADER — cached so it never re-saves artifacts
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_test_data():
    """Load and split data WITHOUT saving artifacts (read-only, cached)."""
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv('data/telco_churn.csv')

    # Clean
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

    # Encode — use saved encoders to be consistent with prediction pipeline
    encoders = joblib.load('models/encoders.pkl')
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)

    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = (df[col] == 'Yes').astype(int)

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col].astype(str))

    # Feature engineering
    df['ChargesPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['TenureGroup'] = pd.cut(
        df['tenure'], bins=[0, 12, 24, 48, 72],
        labels=[0, 1, 2, 3], include_lowest=True
    ).astype(int)
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    df['TotalServices'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v > 0), axis=1
    )

    feature_names = joblib.load('models/feature_names.pkl')
    X = df[feature_names]
    y = df['Churn']

    scaler = joblib.load('models/scaler.pkl')
    X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    y_train = y_train.reset_index(drop=True)
    y_test  = y_test.reset_index(drop=True)
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    render_header()

    # Load dataset & split
    try:
        X_train, X_test, y_train, y_test, feature_names = load_test_data()
        df = pd.read_csv('data/telco_churn.csv')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # Load models
    models = load_all_models(X_train, y_train)

    # ── Sidebar ────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            "<div style='padding: 6px 0 4px; font-size:1.1rem; font-weight:800;"
            " letter-spacing:-0.02em; color:#0f0f0f;'>Churn Predictor</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr style='margin:8px 0 16px;'>", unsafe_allow_html=True)
        st.markdown("### Model")
        model_choice = st.radio(
            "Active model",
            ['XGBoost', 'CatBoost'],
            format_func=lambda x: x,
            label_visibility="collapsed",
        )
        st.markdown("### Threshold")
        threshold = st.slider(
            "Decision threshold",
            0.0, 1.0, 0.5, 0.05,
            help="Probability above which the client is classified as Churn",
        )
        st.markdown("<hr style='margin:16px 0 12px;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.78rem; color:#6b7280; line-height:1.7;'>"
            "<b>Dataset</b> : Telco Customer Churn<br>"
            "<b>Records</b> : 7,043 clients<br>"
            "<b>Features</b> : 21 variables<br>"
            "<b>Course</b> : 4IASD — 2026"
            "</div>",
            unsafe_allow_html=True,
        )

    # 🔴 TABS
    tab_prediction, tab_analytics, tab_insights, tab_optimisation, tab_documentation = st.tabs([
        "Prediction", "Analytics", "Insights", "Optimisation", "Documentation"
    ])

    # ===== TAB 1: PREDICTION =====
    with tab_prediction:
        st.markdown("<div class='ag-section-label'>Single Client Prediction</div>", unsafe_allow_html=True)
        client_data = render_client_form()
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run = st.button("Run Prediction", type="primary", use_container_width=True)
        if run:
            if model_choice in models:
                with st.spinner("Analysing..."):
                    result = predict_churn(client_data, model_choice.lower())
                st.session_state['last_result'] = result
                st.session_state['last_client'] = client_data
            else:
                st.error("No active model loaded.")
        if 'last_result' in st.session_state:
            result = st.session_state['last_result']
            client_save = st.session_state['last_client']
            
            # Re-evaluate with current slider threshold in real time
            proba = result['probability_churn']
            custom_prediction = 1 if proba >= threshold else 0
            result['prediction'] = custom_prediction
            result['label'] = 'CHURN' if custom_prediction == 1 else 'NO CHURN'
            result['risk_level'] = 'Risque eleve' if proba >= 0.70 else 'Risque modere' if proba >= 0.40 else 'Risque faible'
            
            st.markdown("<hr>", unsafe_allow_html=True)
            render_prediction_result(result)
            st.plotly_chart(render_gauge_chart(proba), use_container_width=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            recs = get_retention_recommendations(client_save, proba)
            render_recommendations(recs)
        else:
            st.info("Fill in the form above and click Run Prediction.")

    # ===== TAB 2: ANALYTICS =====
    with tab_analytics:
        st.markdown("<div class='ag-section-label'>Model Performance</div>", unsafe_allow_html=True)
        
        # KPI Row
        render_kpi_row(df)
        st.markdown("<br>", unsafe_allow_html=True)

        if models:
            try:
                from src.evaluation import compute_metrics
                
                # Metrics Table
                st.subheader("Performance Metrics Comparison")
                results = [compute_metrics(m, X_test, y_test, n) for n, m in models.items()]
                df_m = pd.DataFrame(results).set_index('model').round(4)
                st.dataframe(
                    df_m.style.highlight_max(axis=0, color='#dbeafe').format("{:.4f}"),
                    use_container_width=True
                )
                
                st.markdown("---")

                # Confusion Matrix & ROC Curves side-by-side
                col_cm, col_roc = st.columns(2)
                with col_cm:
                    st.subheader("Confusion Matrix")
                    sel_cm = st.selectbox("Select Model for Confusion Matrix", list(models.keys()), key="analytics_cm_select")
                    st.plotly_chart(
                        render_confusion_matrix_plotly(models[sel_cm], X_test, y_test),
                        use_container_width=True
                    )
                with col_roc:
                    st.subheader("ROC Curves")
                    st.plotly_chart(
                        render_roc_plotly(models, X_test, y_test),
                        use_container_width=True
                    )

            except Exception as e:
                import traceback
                st.error(f"Error evaluating models: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
        else:
            st.warning("No models loaded.")

    # ===== TAB 3: INSIGHTS =====
    with tab_insights:
        st.markdown("<div class='ag-section-label'>Insights & Exploration</div>", unsafe_allow_html=True)
        
        if models:
            st.subheader("Feature Importance")
            sel_fi = st.selectbox("Select Model for Feature Importance", ['XGBoost', 'CatBoost'], key="insights_fi_select")
            if sel_fi in models:
                st.plotly_chart(
                    render_feature_importance_plotly(models[sel_fi], feature_names, sel_fi.upper()),
                    use_container_width=True
                )
            st.markdown("---")
            
        # Exploratory Data Analysis & Segments
        st.subheader("Interactive Exploration")
        contract_filter = st.multiselect(
            "Filter by contract", df['Contract'].unique(),
            default=list(df['Contract'].unique()),
            key="insights_contract_filter"
        )
        internet_filter = st.multiselect(
            "Filter by Internet service", df['InternetService'].unique(),
            default=list(df['InternetService'].unique()),
            key="insights_internet_filter"
        )

        df_filtered = df[
            (df['Contract'].isin(contract_filter)) &
            (df['InternetService'].isin(internet_filter))
        ]

        if df_filtered.empty:
            st.warning("No data matches selected filters.")
        else:
            render_financial_analysis(df_filtered)
            st.markdown("---")
            render_demographic_analysis(df_filtered)
            st.markdown("---")
            render_service_analysis(df_filtered)

        st.markdown("---")
        st.subheader("Risk Segments Analysis")
        render_funnel_analysis(df)
        
        st.markdown("#### Strategic Retention Recommendations by Segment")
        st.markdown("""
        - **Month-to-month Contract + Fiber Optic**: This segment has a very high churn rate (> 50%). **Action:** Aggressively offer a 1-year contract with a 10% discount.
        - **Seniors on Month-to-month Contract**: **Action:** Proactive customer support check-in calls to ensure satisfaction and offer tech assistance.
        - **Electronic Check Payments**: **Action:** Incentivize moving to automatic credit card or bank transfer by offering a one-time account credit.
        """)

    # ===== TAB 4: OPTIMISATION =====
    with tab_optimisation:
        st.markdown("<div class='ag-section-label'>Hyperparameter Optimisation</div>", unsafe_allow_html=True)
        st.markdown(
            "Launch hyperparameter optimization to improve model performance. "
            "The best model found will be saved and compared to the baseline model."
        )
        st.markdown("---")

        c1, c2, c3 = st.columns(3)
        with c1:
            method = st.selectbox("Optimization Method", ['RandomizedSearch', 'Optuna'], key="opt_method")
        with c2:
            model_type = st.selectbox("Model Type", ['xgboost', 'catboost'], key="opt_model_type")
        with c3:
            n_iter = st.slider("Iterations / Trials", 10, 100, 30, 10, key="opt_n_iter")

        if method == 'Optuna':
            st.info("Optuna uses Bayesian optimization (TPE) - more efficient than RandomizedSearch.")
        else:
            st.info("RandomizedSearch samples the hyperparameter space randomly over the chosen iterations.")

        if st.button("Launch Optimisation", type="primary", key="opt_launch_btn"):
            from src.hyperparameter_tuning import (
                tune_xgboost_random,
                tune_catboost_random,
                tune_with_optuna,
            )
            from src.evaluation import compute_metrics

            with st.spinner(f"Optimisation in progress ({method}, {n_iter} iterations) - this may take a few minutes..."):
                try:
                    X_train, X_test, y_train, y_test, features = load_test_data()
                    X_train = X_train.copy()
                    y_train = y_train.copy()

                    if method == 'RandomizedSearch':
                        best = (
                            tune_xgboost_random(X_train, y_train, n_iter=n_iter)
                            if model_type == 'xgboost'
                            else tune_catboost_random(X_train, y_train, n_iter=n_iter)
                        )
                    else:
                        best = tune_with_optuna(X_train, y_train, model_type, n_trials=n_iter)

                    if best is not None:
                        st.success("Optimisation complete - optimized model saved.")

                        # Before/after comparison
                        baseline = models.get('XGBoost' if model_type == 'xgboost' else 'CatBoost')
                        comparison_models = {}
                        if baseline:
                            comparison_models[f"{model_type.upper()} (baseline)"] = baseline
                        comparison_models[f"{model_type.upper()} ({method})"] = best

                        st.markdown("### Before / After Performance Comparison")
                        results = [
                            compute_metrics(m, X_test, y_test, name)
                            for name, m in comparison_models.items()
                        ]
                        df_comp = pd.DataFrame(results).set_index('model').round(4)
                        st.dataframe(
                            df_comp.style.highlight_max(axis=0, color='#dbeafe').format("{:.4f}"),
                            use_container_width=True
                        )

                        st.markdown("### Best Hyperparameters")
                        st.json(best.get_params())
                    else:
                        st.error("Optimisation failed to complete.")

                except Exception as e:
                    import traceback
                    st.error(f"Error during optimization: {e}")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())

    # ===== TAB 5: DOCUMENTATION =====
    with tab_documentation:
        st.markdown("<div class='ag-section-label'>Context & Objectives</div>", unsafe_allow_html=True)
        st.markdown(
            """
            ### Welcome to Churn Predictor
            This application has been designed to help business teams proactively identify customers at risk of churn before they cancel their subscriptions.

            #### Objectives
            - **Anticipate**: Predict the churn risk of any customer in real time using Machine Learning.
            - **Understand**: Discover the key driving factors behind customer churn through interactive visualization and segments analysis.
            - **Act**: Take action with customized retention recommendations designed for each high-risk customer profile.

            #### How to use this application?
            Navigate through the tabs above to access different sections of the tool:
            - **Prediction**: Simulate a customer profile in real time to obtain their churn probability and retention suggestions.
            - **Analytics**: Evaluate model performances, view metrics, confusion matrices, and ROC curves.
            - **Insights**: Explore customer demographic distribution, services, and high-churn risk segments.
            - **Optimisation**: Fine-tune XGBoost or CatBoost models using randomized search or Optuna Bayesian search.
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()