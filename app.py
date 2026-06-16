# app.py
import streamlit as st
import joblib
import pandas as pd
import os
import plotly.express as px

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
    apply_custom_css, render_header, render_client_form,
    render_prediction_result, render_gauge_chart,
    render_confusion_matrix_plotly, render_roc_plotly,
    render_feature_importance_plotly, render_recommendations,
)

apply_custom_css()


# ─────────────────────────────────────────────────────────────
#  MODEL LOADER
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models, feature_names = {}, []
    try:
        models['xgboost']  = joblib.load('models/xgb_model.pkl')
        models['catboost'] = joblib.load('models/cat_model.pkl')
        feature_names      = joblib.load('models/feature_names.pkl')
    except FileNotFoundError:
        st.error("Models not found. Run the training script first.")
    return models, feature_names


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    render_header()
    models, feature_names = load_models()

    # ── Sidebar ──────────────────────────────────────────────
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
            ['xgboost', 'catboost'],
            format_func=lambda x: "XGBoost" if x == 'xgboost' else "CatBoost",
            label_visibility="collapsed",
        )

        st.markdown("### Threshold")
        threshold = st.slider(
            "Decision threshold", 0.0, 1.0, 0.5, 0.05,
            help="Probability above which the client is classified as Churn",
            label_visibility="visible",
        )

        st.markdown("### Navigation")
        page = st.radio(
            "Page",
            [
                "Prediction",
                "Analytics Dashboard",
                "Model Comparison",
                "Hyperparameter Tuning",
                "Batch Prediction (CSV)",
            ],
            label_visibility="collapsed",
        )

        st.markdown("<hr style='margin:16px 0 12px;'>", unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.78rem; color:#6b7280; line-height:1.7;'>"
            "<b>Dataset</b>: Telco Customer Churn<br>"
            "<b>Records</b>: 7,043 clients<br>"
            "<b>Features</b>: 21 variables<br>"
            "<b>Course</b>: 4IASD — 2026"
            "</div>",
            unsafe_allow_html=True,
        )

    # ══════════════════════════════════════════════════════════
    #  PAGE 1 — PREDICTION
    # ══════════════════════════════════════════════════════════
    if page == "Prediction":
        st.markdown(
            "<div class='ag-section-label'>Single Client Prediction</div>",
            unsafe_allow_html=True,
        )
        client_data = render_client_form()
        col_btn, _ = st.columns([1, 3])
        with col_btn:
            run = st.button("Run Prediction", type="primary", use_container_width=True)

        if run:
            if models:
                with st.spinner("Analysing…"):
                    result = predict_churn(client_data, model_choice)
                    st.session_state['last_result'] = result
                    st.session_state['last_client'] = client_data
            else:
                st.error("No models loaded.")

        if 'last_result' in st.session_state:
            result      = st.session_state['last_result']
            client_save = st.session_state['last_client']

            st.markdown("<hr>", unsafe_allow_html=True)
            render_prediction_result(result)
            st.plotly_chart(
                render_gauge_chart(result['probability_churn']),
                use_container_width=True,
            )
            st.markdown("<hr>", unsafe_allow_html=True)
            recs = get_retention_recommendations(client_save, result['probability_churn'])
            render_recommendations(recs)

        else:
            st.info("Fill in the form above and click Run Prediction.")

    # ══════════════════════════════════════════════════════════
    #  PAGE 2 — ANALYTICS DASHBOARD
    # ══════════════════════════════════════════════════════════
    elif page == "Analytics Dashboard":
        try:
            df = pd.read_csv('data/telco_churn.csv')
            from src.dashboard import render_dashboard
            render_dashboard(df)
        except FileNotFoundError:
            st.error("Dataset not found. Place telco_churn.csv in the data/ folder.")
        except Exception as e:
            st.error(f"Error: {e}")

    # ══════════════════════════════════════════════════════════
    #  PAGE 3 — MODEL COMPARISON
    # ══════════════════════════════════════════════════════════
    elif page == "Model Comparison":
        st.markdown(
            "<div class='ag-section-label'>7-Model Benchmark</div>",
            unsafe_allow_html=True,
        )

        if models:
            try:
                from src.preprocessing import prepare_data
                from src.evaluation import compute_metrics

                with st.spinner("Preparing data…"):
                    X_train, X_test, y_train, y_test, features, _ = prepare_data(
                        'data/telco_churn.csv'
                    )

                tab1, tab2, tab3 = st.tabs(["Metrics", "Confusion Matrix", "ROC Curves"])

                with tab1:
                    results = [compute_metrics(m, X_test, y_test, n) for n, m in models.items()]
                    df_m = pd.DataFrame(results).set_index('model').round(4)
                    st.dataframe(
                        df_m.style.highlight_max(axis=0, color='#dbeafe').format("{:.4f}"),
                        use_container_width=True,
                    )
                    st.markdown("**Feature Importance**")
                    sel = st.selectbox("Model", list(models.keys()), key="fi_select")
                    st.plotly_chart(
                        render_feature_importance_plotly(models[sel], features, sel.upper()),
                        use_container_width=True,
                    )
                with tab2:
                    sel2 = st.selectbox("Model", list(models.keys()), key="cm_select")
                    st.plotly_chart(
                        render_confusion_matrix_plotly(models[sel2], X_test, y_test),
                        use_container_width=True,
                    )
                with tab3:
                    st.plotly_chart(
                        render_roc_plotly(models, X_test, y_test),
                        use_container_width=True,
                    )
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("No models loaded.")

    # ══════════════════════════════════════════════════════════
    #  PAGE 4 — HYPERPARAMETER TUNING
    # ══════════════════════════════════════════════════════════
    elif page == "Hyperparameter Tuning":
        st.markdown(
            "<div class='ag-section-label'>Hyperparameter Optimisation</div>",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            method     = st.selectbox("Method", ['RandomizedSearch', 'Optuna'])
        with c2:
            model_type = st.selectbox("Model", ['xgboost', 'catboost'])
        with c3:
            n_iter = st.slider("Iterations / Trials", 10, 100, 30, 10)
        if st.button("Start Optimisation", type="primary"):
            from src.preprocessing import prepare_data
            from src.hyperparameter_tuning import (
                tune_xgboost_random, tune_catboost_random, tune_with_optuna
            )
            with st.spinner("Optimisation in progress — this may take several minutes…"):
                X_train, X_test, y_train, y_test, features, _ = prepare_data(
                    'data/telco_churn.csv'
                )
                if method == 'RandomizedSearch':
                    best = (tune_xgboost_random(X_train, y_train, n_iter=n_iter)
                            if model_type == 'xgboost'
                            else tune_catboost_random(X_train, y_train, n_iter=n_iter))
                else:
                    best = tune_with_optuna(X_train, y_train, model_type, n_trials=n_iter)

            if best is not None:
                st.success("Optimisation complete. Model saved.")
                st.markdown("**Best Parameters**")
                st.json(best.get_params())
            else:
                st.error("Optimisation could not be completed.")

    # ══════════════════════════════════════════════════════════
    #  PAGE 5 — BATCH PREDICTION
    # ══════════════════════════════════════════════════════════
    elif page == "Batch Prediction (CSV)":
        st.markdown(
            "<div class='ag-section-label'>Batch Prediction</div>",
            unsafe_allow_html=True,
        )
        st.info("Upload a CSV file with the same columns as the Telco Customer Churn dataset.")
        c1, c2 = st.columns([2, 1])
        with c1:
            uploaded = st.file_uploader("Select CSV file", type=['csv'])
        with c2:
            model_batch = st.selectbox(
                "Model",
                ['xgboost', 'catboost'],
                format_func=lambda x: "XGBoost" if x == 'xgboost' else "CatBoost",
                key="batch_model",
            )
        if uploaded:
            df_up = pd.read_csv(uploaded)
            st.markdown(f"**Preview** — {len(df_up):,} rows × {len(df_up.columns)} columns")
            st.dataframe(df_up.head(), use_container_width=True)
            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner(f"Running predictions on {len(df_up):,} clients…"):
                    results, errors = [], []
                    for idx, row in df_up.iterrows():
                        try:
                            results.append(predict_churn(row.to_dict(), model_batch))
                        except Exception as e:
                            errors.append(f"Row {idx}: {e}")

                df_res = pd.DataFrame(results)
                st.markdown(f"**Results** — {len(df_res):,} predictions")
                st.dataframe(df_res, use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    df_res.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
                if errors:
                    with st.expander(f"{len(errors)} rows failed — click to expand"):
                        for e in errors:
                            st.text(e)


if __name__ == "__main__":
    main()