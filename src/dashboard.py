"""
Page Dashboard Analytique complète pour Streamlit.
Importer et appeler render_dashboard(df, models, model_choice, threshold) dans app.py.
"""
import pandas as pd
import streamlit as st
import plotly.express as px


def render_kpi_row(df: pd.DataFrame):
    """Ligne de KPIs en haut du dashboard."""
    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    df_clean['Churn_bin'] = (df_clean['Churn'] == 'Yes').astype(int)

    churn_rate = df_clean['Churn_bin'].mean()
    avg_tenure = df_clean['tenure'].mean()
    avg_monthly = df_clean['MonthlyCharges'].mean()
    revenue_at_risk = (
        df_clean[df_clean['Churn'] == 'Yes']['MonthlyCharges'].sum()
    )

    cols = st.columns(4)
    cols[0].metric(
        "Total Clients", f"{len(df_clean):,}",
        help="Taille totale du dataset"
    )
    cols[1].metric(
        "Taux de Churn", f"{churn_rate:.1%}",
        delta=f"{churn_rate - 0.265:.1%} vs ref 26.5%",
        delta_color="inverse"
    )
    cols[2].metric(
        "Durée Moyenne", f"{avg_tenure:.0f} mois",
        help="Tenure moyen de tous les clients"
    )
    cols[3].metric(
        "Revenu à Risque", f"${revenue_at_risk:,.0f}/mois",
        help="Charges mensuelles des clients en churn",
        delta_color="inverse"
    )


def render_churn_overview(df: pd.DataFrame):
    """Section : vue d'ensemble du churn."""
    st.subheader("Vue d'Ensemble du Churn")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        counts = df['Churn'].value_counts()
        fig = px.pie(
            values=counts.values, names=counts.index,
            title="Répartition Churn / No Churn",
            color=counts.index,
            color_discrete_map={'Yes': '#111827', 'No': '#6b7280'},
            hole=0.45
        )
        fig.update_traces(textinfo='percent+label', textfont_size=13)
        fig.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        contract_df = df.groupby(['Contract', 'Churn']).size().reset_index(name='n')
        contract_total = df.groupby('Contract').size().reset_index(name='total')
        contract_df = contract_df.merge(contract_total, on='Contract')
        contract_df['rate'] = contract_df['n'] / contract_df['total']

        churn_only = contract_df[contract_df['Churn'] == 'Yes']
        fig2 = px.bar(
            churn_only, x='Contract', y='rate',
            title="Taux de Churn par Contrat",
            color='rate', color_continuous_scale='Greys',
            text=churn_only['rate'].apply(lambda x: f"{x:.0%}"),
            range_color=[0, 0.5]
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(
            height=320, margin=dict(t=40, b=20),
            coloraxis_showscale=False, yaxis_tickformat='.0%'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        internet_df = (
            df.groupby(['InternetService', 'Churn'])
            .size().reset_index(name='n')
        )
        fig3 = px.bar(
            internet_df, x='InternetService', y='n',
            color='Churn', barmode='group',
            title="Distribution par Service Internet",
            color_discrete_map={'Yes': '#111827', 'No': '#6b7280'},
            text='n'
        )
        fig3.update_traces(textposition='outside')
        fig3.update_layout(height=320, margin=dict(t=40, b=20))
        st.plotly_chart(fig3, use_container_width=True)


def render_financial_analysis(df: pd.DataFrame):
    """Section : analyse financière."""
    st.subheader("Analyse Financière")

    df_clean = df.copy()
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

    col1, col2 = st.columns(2)

    with col1:
        fig = px.violin(
            df_clean, x='Churn', y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'Yes': '#111827', 'No': '#6b7280'},
            title="Distribution des Charges Mensuelles",
            box=True, points='outliers'
        )
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        sample = df_clean.sample(min(1000, len(df_clean)), random_state=42)
        fig2 = px.scatter(
            sample, x='tenure', y='MonthlyCharges',
            color='Churn',
            color_discrete_map={'Yes': '#111827', 'No': '#6b7280'},
            title="Durée vs Charges (échantillon 1000 clients)",
            opacity=0.6, size_max=6
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)


def render_demographic_analysis(df: pd.DataFrame):
    """Section : profil démographique des churners."""
    st.subheader("Profil Démographique des Churners")

    df = df.copy()
    col1, col2, col3 = st.columns(3)

    with col1:
        gender_churn = (
            df.groupby(['gender', 'Churn']).size() /
            df.groupby('gender').size()
        ).reset_index(name='rate').query("Churn == 'Yes'")
        fig = px.bar(
            gender_churn, x='gender', y='rate',
            title="Taux de Churn par Genre",
            color='gender', text=gender_churn['rate'].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=['#111827', '#6b7280']
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(height=300, yaxis_tickformat='.0%', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        senior_map = {0: 'Non-Senior', 1: 'Senior'}
        df['SeniorLabel'] = df['SeniorCitizen'].map(senior_map)
        senior_churn = (
            df.groupby(['SeniorLabel', 'Churn']).size() /
            df.groupby('SeniorLabel').size()
        ).reset_index(name='rate').query("Churn == 'Yes'")
        fig2 = px.bar(
            senior_churn, x='SeniorLabel', y='rate',
            title="Taux de Churn : Senior vs Non-Senior",
            color='SeniorLabel',
            text=senior_churn['rate'].apply(lambda x: f"{x:.1%}"),
            color_discrete_sequence=['#111827', '#6b7280']
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(height=300, yaxis_tickformat='.0%', showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        df['TenureGroup'] = pd.cut(
            df['tenure'],
            bins=[0, 12, 24, 48, 72],
            labels=['0-12m', '12-24m', '24-48m', '48-72m'],
            include_lowest=True
        )
        heat = (
            df[df['Churn'] == 'Yes']
            .groupby(['TenureGroup', 'Contract']).size()
            .unstack(fill_value=0)
        )
        fig3 = px.imshow(
            heat, title="Churners : Tenure x Contrat",
            color_continuous_scale='Greys', text_auto=True
        )
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)


def render_service_analysis(df: pd.DataFrame):
    """Section : analyse des services souscrits."""
    st.subheader("Analyse des Services")

    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]

    service_churn = []
    for col in service_cols:
        if col not in df.columns:
            continue
        for val in df[col].unique():
            if val in ['No internet service', 'No phone service']:
                continue
            mask = df[col] == val
            rate = (df.loc[mask, 'Churn'] == 'Yes').mean()
            service_churn.append({
                'Service': col, 'Valeur': val,
                'Taux Churn': rate, 'N': int(mask.sum())
            })

    df_svc = pd.DataFrame(service_churn)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        fig = px.bar(
            df_svc, x='Service', y='Taux Churn',
            color='Valeur', barmode='group',
            title="Taux de Churn selon l'Abonnement aux Services",
            text=df_svc['Taux Churn'].apply(lambda x: f"{x:.0%}"),
            color_discrete_sequence=['#111827', '#6b7280']
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            height=420, yaxis_tickformat='.0%',
            xaxis_tickangle=-30
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        pay_churn = df.groupby(['PaymentMethod', 'Churn']).size().reset_index(name='n')
        fig2 = px.treemap(
            pay_churn, path=['PaymentMethod', 'Churn'], values='n',
            title="Répartition par Méthode de Paiement",
            color='Churn',
            color_discrete_map={'Yes': '#111827', 'No': '#6b7280'}
        )
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)


def render_funnel_analysis(df: pd.DataFrame):
    """Section : analyse en entonnoir — segments à risque."""
    st.subheader("Segments Clients à Risque")

    segments = {
        'Contrat mensuel': (df['Contract'] == 'Month-to-month'),
        '+ Fibre optique':
            (df['Contract'] == 'Month-to-month') &
            (df['InternetService'] == 'Fiber optic'),
        '+ Senior':
            (df['Contract'] == 'Month-to-month') &
            (df['InternetService'] == 'Fiber optic') &
            (df['SeniorCitizen'] == 1),
        '+ Paiement chèque':
            (df['Contract'] == 'Month-to-month') &
            (df['InternetService'] == 'Fiber optic') &
            (df['PaymentMethod'] == 'Electronic check'),
        '+ < 12 mois':
            (df['Contract'] == 'Month-to-month') &
            (df['InternetService'] == 'Fiber optic') &
            (df['tenure'] < 12),
    }

    funnel_data = []
    for label, mask in segments.items():
        subset = df[mask]
        churn_rate = (subset['Churn'] == 'Yes').mean()
        funnel_data.append({
            'Segment': label,
            'N Clients': int(mask.sum()),
            'Taux Churn': churn_rate,
            'N Churners': int((subset['Churn'] == 'Yes').sum())
        })

    df_funnel = pd.DataFrame(funnel_data)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.funnel(
            df_funnel, x='N Clients', y='Segment',
            title="Entonnoir des Segments à Risque"
        )
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            df_funnel, x='Segment', y='Taux Churn',
            color='Taux Churn',
            color_continuous_scale='Greys',
            text=df_funnel['Taux Churn'].apply(lambda x: f"{x:.1%}"),
            title="Taux de Churn par Segment"
        )
        fig2.update_traces(textposition='outside')
        fig2.update_layout(
            height=380, yaxis_tickformat='.0%',
            coloraxis_showscale=False,
            xaxis_tickangle=-20
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.dataframe(
        df_funnel.style.format({
            'Taux Churn': '{:.1%}',
            'N Clients': '{:,}',
            'N Churners': '{:,}'
        }).background_gradient(subset=['Taux Churn'], cmap='Greys'),
        use_container_width=True
    )


def render_model_performance_tab(models: dict, model_choice: str):
    """
    Tab 3 — Performance du Modèle.
    Affiche confusion matrix, ROC curves et feature importance à partir des modèles chargés.
    """
    from src.ui_components import (
        render_confusion_matrix_plotly,
        render_roc_plotly,
        render_feature_importance_plotly,
    )
    from src.evaluation import compute_metrics

    if not models:
        st.error("Aucun modèle chargé. Vérifiez que les fichiers .pkl existent dans models/.")
        return

    # Load test data (cached via app.py helper, but we import it here safely)
    try:
        import joblib
        import numpy as np
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        df_raw = pd.read_csv('data/telco_churn.csv')
        if 'customerID' in df_raw.columns:
            df_raw.drop('customerID', axis=1, inplace=True)
        df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
        df_raw['TotalCharges'] = df_raw['TotalCharges'].fillna(df_raw['TotalCharges'].median())

        encoders = joblib.load('models/encoders.pkl')
        df_raw['Churn'] = (df_raw['Churn'] == 'Yes').astype(int)

        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen']
        for col in binary_cols:
            if col in df_raw.columns and df_raw[col].dtype == object:
                df_raw[col] = (df_raw[col] == 'Yes').astype(int)

        for col, le in encoders.items():
            if col in df_raw.columns:
                df_raw[col] = le.transform(df_raw[col].astype(str))

        df_raw['ChargesPerMonth'] = df_raw['TotalCharges'] / (df_raw['tenure'] + 1)
        df_raw['TenureGroup'] = pd.cut(
            df_raw['tenure'], bins=[0, 12, 24, 48, 72],
            labels=[0, 1, 2, 3], include_lowest=True
        ).astype(int)
        service_cols = [
            'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        df_raw['TotalServices'] = df_raw[service_cols].apply(
            lambda row: sum(1 for v in row if v > 0), axis=1
        )

        feature_names = joblib.load('models/feature_names.pkl')
        X = df_raw[feature_names]
        y = df_raw['Churn']
        scaler = joblib.load('models/scaler.pkl')
        X_scaled = pd.DataFrame(scaler.transform(X), columns=feature_names)
        _, X_test, _, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        y_test = y_test.reset_index(drop=True)

    except Exception as e:
        st.error(f"Erreur lors du chargement des données de test : {e}")
        return

    # ── KPIs rapides ──────────────────────────────────────
    st.subheader("Métriques de Performance")
    results = [compute_metrics(m, X_test, y_test, n) for n, m in models.items()]
    df_metrics = pd.DataFrame(results).set_index('model').round(4)
    st.dataframe(
        df_metrics.style.highlight_max(axis=0, color='#dbeafe').format("{:.4f}"),
        use_container_width=True
    )

    st.markdown("---")

    # ── Confusion Matrix + ROC côte à côte ────────────────
    col_cm, col_roc = st.columns(2)
    with col_cm:
        sel_cm = st.selectbox("Modèle — Matrice de Confusion", list(models.keys()), key="dash_cm")
        st.plotly_chart(
            render_confusion_matrix_plotly(models[sel_cm], X_test, y_test),
            use_container_width=True
        )
    with col_roc:
        st.plotly_chart(
            render_roc_plotly(models, X_test, y_test),
            use_container_width=True
        )

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────
    st.subheader("Importance des Variables")
    sel_fi = st.selectbox("Modèle — Feature Importance", list(models.keys()), key="dash_fi")
    st.plotly_chart(
        render_feature_importance_plotly(models[sel_fi], feature_names, sel_fi.upper()),
        use_container_width=True
    )


def render_individual_prediction_tab(models: dict, model_choice: str, threshold: float):
    """
    Tab 4 — Prédiction Individuelle intégrée dans le dashboard.
    Formulaire complet avec résultat et recommandations.
    """
    from src.ui_components import (
        render_client_form,
        render_prediction_result,
        render_gauge_chart,
        render_recommendations,
    )
    from src.prediction import predict_churn, get_retention_recommendations

    if not models:
        st.error("Aucun modèle chargé.")
        return

    st.subheader("Prédiction Individuelle — Client")
    st.caption(
        f"Modèle actif : **{model_choice.upper()}** · "
        f"Seuil de décision : **{threshold:.0%}**"
    )
    st.markdown("---")

    # Use a unique key prefix so this form is independent from the sidebar page
    client_data = render_client_form()

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run = st.button("Analyser ce client", type="primary",
                        use_container_width=True, key="dash_pred_btn")

    if run:
        with st.spinner("Analyse en cours…"):
            result = predict_churn(client_data, model_choice)
        st.session_state['dash_result'] = result
        st.session_state['dash_client'] = client_data

    if 'dash_result' in st.session_state:
        result = st.session_state['dash_result']
        client_save = st.session_state['dash_client']
        st.markdown("<hr>", unsafe_allow_html=True)
        render_prediction_result(result)
        st.plotly_chart(render_gauge_chart(result['probability_churn']), use_container_width=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        recs = get_retention_recommendations(client_save, result['probability_churn'])
        render_recommendations(recs)
    else:
        st.info("Remplissez le formulaire ci-dessus et cliquez sur Analyser.")


def render_dashboard(df: pd.DataFrame, models: dict, model_choice: str, threshold: float):
    """
    Dashboard analytique complet structuré par onglets.
    Appeler cette fonction depuis app.py.
    """
    st.title("Dashboard Analytique — Churn Client")

    # KPIs en haut (toujours visibles)
    render_kpi_row(df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Onglets
    tabs = st.tabs([
        "1. Overview",
        "2. Exploration",
        "3. Performance du Modèle",
        "4. Prédiction Individuelle",
        "5. Segments à Risque"
    ])

    with tabs[0]:
        render_churn_overview(df)

    with tabs[1]:
        st.subheader("Exploration Interactive")
        contract_filter = st.multiselect(
            "Filtrer par contrat", df['Contract'].unique(),
            default=list(df['Contract'].unique())
        )
        internet_filter = st.multiselect(
            "Filtrer par service Internet", df['InternetService'].unique(),
            default=list(df['InternetService'].unique())
        )

        df_filtered = df[
            (df['Contract'].isin(contract_filter)) &
            (df['InternetService'].isin(internet_filter))
        ]

        if df_filtered.empty:
            st.warning("Aucune donnée avec ces filtres.")
        else:
            render_financial_analysis(df_filtered)
            st.markdown("---")
            render_demographic_analysis(df_filtered)
            st.markdown("---")
            render_service_analysis(df_filtered)

    with tabs[2]:
        render_model_performance_tab(models, model_choice)

    with tabs[3]:
        render_individual_prediction_tab(models, model_choice, threshold)

    with tabs[4]:
        render_funnel_analysis(df)
        st.markdown("#### Recommandations Stratégiques par Segment")
        st.markdown("""
        - **Contrat Mensuel + Fibre** : Ce segment présente un risque de fuite très élevé (> 50%). **Action :** Proposer agressivement un renouvellement d'un an avec 10% de remise.
        - **Seniors sur Contrat Mensuel** : **Action :** Un appel du support technique pour vérifier la satisfaction globale peut réduire la frustration liée à la technologie.
        - **Paiements par chèque électronique** : **Action :** Simplifier la transition vers la carte de crédit automatique en offrant un mois gratuit de "Device Protection".
        """)
