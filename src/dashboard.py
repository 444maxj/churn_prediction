"""
Page Dashboard Analytique complète pour Streamlit.
Importer et appeler render_dashboard(df) dans app.py.
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
            heat, title="Churners : Tenure × Contrat",
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


def render_dashboard(df: pd.DataFrame):
    """
    Dashboard analytique complet.
    Appeler cette fonction depuis app.py.
    """
    st.title("Dashboard Analytique — Churn Client")
    st.markdown("---")

    render_kpi_row(df)
    st.markdown("---")
    render_churn_overview(df)
    st.markdown("---")
    render_financial_analysis(df)
    st.markdown("---")
    render_demographic_analysis(df)
    st.markdown("---")
    render_service_analysis(df)
    st.markdown("---")
    render_funnel_analysis(df)
