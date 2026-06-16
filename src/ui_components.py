# src/ui_components.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix


# ─────────────────────────────────────────────────────────────
#  DESIGN TOKENS
#  Palette: Warm canvas · Ink black · Electric blue accent
# ─────────────────────────────────────────────────────────────
ACCENT       = "#1d4ed8"   # electric blue
ACCENT_SOFT  = "#dbeafe"   # blue-100
INK          = "#0f0f0f"   # near-black
INK_2        = "#374151"   # cool slate
INK_3        = "#6b7280"   # muted label
CANVAS       = "#f5f4f0"   # warm off-white (app bg)
SURFACE      = "#ffffff"   # pure white cards
BORDER       = "#e5e7eb"   # dividers
DANGER       = "#dc2626"   # churn red
SUCCESS      = "#16a34a"   # no-churn green


def apply_custom_css():
    st.markdown(f"""
    <style>
    /* ── Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    }}

    /* ── Canvas background ── */
    .stApp {{
        background-color: {CANVAS} !important;
        background-image:
            radial-gradient(circle at 10% 20%, rgba(29,78,216,0.04) 0%, transparent 55%),
            radial-gradient(circle at 90% 80%, rgba(29,78,216,0.03) 0%, transparent 50%);
    }}

    /* ── Typography ── */
    h1, h2, h3, h4, h5 {{
        font-family: 'Inter', sans-serif !important;
        color: {INK} !important;
        letter-spacing: -0.025em !important;
        line-height: 1.2 !important;
    }}
    h1 {{ font-size: 2rem !important; font-weight: 800 !important; }}
    h2 {{ font-size: 1.5rem !important; font-weight: 700 !important; }}
    h3 {{ font-size: 1.15rem !important; font-weight: 700 !important; }}
    p, li, span {{ color: {INK_2}; line-height: 1.65; }}

    /* ── Cards ── */
    .ag-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 24px 28px;
        margin-bottom: 18px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 4px 16px rgba(0,0,0,0.04);
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }}
    .ag-card:hover {{
        box-shadow: 0 4px 20px rgba(29,78,216,0.10), 0 1px 4px rgba(0,0,0,0.06);
        transform: translateY(-2px);
    }}

    /* ── Section header chip ── */
    .ag-section-label {{
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {ACCENT};
        background: {ACCENT_SOFT};
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        margin-bottom: 14px;
    }}

    /* ── Dividers ── */
    hr {{
        border: none !important;
        border-top: 1px solid {BORDER} !important;
        margin: 24px 0 !important;
    }}

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {{
        background-color: {SURFACE} !important;
        border-right: 1px solid {BORDER} !important;
    }}
    section[data-testid="stSidebar"] .stMarkdown h3 {{
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: {INK_3} !important;
        margin-top: 20px !important;
        margin-bottom: 6px !important;
    }}

    /* ── Radio nav pills ── */
    div[data-testid="stRadio"] div[role="radiogroup"] label {{
        border-radius: 9px !important;
        padding: 9px 14px !important;
        margin-bottom: 4px !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: {INK_2} !important;
        border: 1px solid transparent !important;
        transition: all 0.18s ease !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label:hover {{
        background: rgba(29,78,216,0.06) !important;
        color: {ACCENT} !important;
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] {{
        background: {ACCENT_SOFT} !important;
        color: {ACCENT} !important;
        font-weight: 600 !important;
        border-color: rgba(29,78,216,0.2) !important;
    }}

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 18px 20px !important;
        transition: all 0.2s ease !important;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: rgba(29,78,216,0.25) !important;
        box-shadow: 0 0 0 3px rgba(29,78,216,0.06) !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: 0.75rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        color: {INK_3} !important;
    }}
    div[data-testid="stMetricValue"] {{
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: {INK} !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: -0.03em !important;
    }}
    div[data-testid="stMetricDelta"] {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }}

    /* ── Primary button ── */
    div.stButton > button[kind="primary"] {{
        background: {INK} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 11px 28px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.015em !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(15,15,15,0.18) !important;
    }}
    div.stButton > button[kind="primary"]:hover {{
        background: {ACCENT} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(29,78,216,0.3) !important;
    }}
    div.stButton > button[kind="secondary"] {{
        background: {SURFACE} !important;
        color: {INK} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}
    div.stButton > button[kind="secondary"]:hover {{
        border-color: {ACCENT} !important;
        color: {ACCENT} !important;
        background: {ACCENT_SOFT} !important;
    }}

    /* ── Form inputs ── */
    .stSelectbox label, .stNumberInput label, .stSlider label,
    .stTextInput label, .stFileUploader label {{
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: {INK_2} !important;
        letter-spacing: 0.01em !important;
    }}
    div[data-baseweb="select"] > div {{
        border-radius: 9px !important;
        border-color: {BORDER} !important;
        background: {SURFACE} !important;
        transition: border-color 0.18s !important;
    }}
    div[data-baseweb="select"] > div:focus-within {{
        border-color: {ACCENT} !important;
        box-shadow: 0 0 0 3px rgba(29,78,216,0.1) !important;
    }}

    /* ── Tabs ── */
    div[data-testid="stTabs"] button[role="tab"] {{
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: {INK_3} !important;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
    }}
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
        color: {ACCENT} !important;
        border-bottom: 2px solid {ACCENT} !important;
    }}

    /* ── Info / Warning / Success banners ── */
    div[data-testid="stInfo"] {{
        background: {ACCENT_SOFT} !important;
        border: 1px solid rgba(29,78,216,0.2) !important;
        border-radius: 10px !important;
        color: {ACCENT} !important;
    }}

    /* ── Status card variants ── */
    .ag-status-churn {{
        border-left: 4px solid {DANGER} !important;
    }}
    .ag-status-ok {{
        border-left: 4px solid {SUCCESS} !important;
    }}

    /* ── Recommendation item ── */
    .ag-rec-item {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        font-weight: 500;
        color: {INK_2};
        transition: border-color 0.2s, box-shadow 0.2s;
    }}
    .ag-rec-item:hover {{
        border-left-color: {INK};
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
    }}

    /* ── Group label inside form ── */
    .ag-group-label {{
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: {INK_3};
        border-bottom: 1px solid {BORDER};
        padding-bottom: 6px;
        margin-top: 18px;
        margin-bottom: 12px;
    }}

    /* ── Fade-in animation ── */
    @keyframes agFadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .ag-animate {{
        animation: agFadeUp 0.35s cubic-bezier(0.22, 1, 0.36, 1) both;
    }}
    .ag-delay-1 {{ animation-delay: 0.05s; }}
    .ag-delay-2 {{ animation-delay: 0.12s; }}
    .ag-delay-3 {{ animation-delay: 0.20s; }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 6px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 99px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #9ca3af; }}

    /* ── Header orbital (CSS-div rings, no SVG) ── */
    .ag-orbital {{
        position: relative;
        width: 120px; height: 120px;
        flex-shrink: 0;
    }}
    .ag-orbital-core {{
        position: absolute;
        top: 50%; left: 50%;
        width: 22px; height: 22px;
        margin: -11px 0 0 -11px;
        background: linear-gradient(135deg, {ACCENT} 0%, #6366f1 100%);
        border-radius: 50%;
        box-shadow: 0 0 18px rgba(29,78,216,0.45);
        animation: ag-core-pulse 3s ease-in-out infinite alternate;
    }}
    @keyframes ag-core-pulse {{
        from {{ box-shadow: 0 0 12px rgba(29,78,216,0.35); transform: scale(0.95); }}
        to   {{ box-shadow: 0 0 26px rgba(29,78,216,0.60); transform: scale(1.05); }}
    }}
    .ag-ring {{
        position: absolute;
        top: 50%; left: 50%;
        border: 1.5px solid rgba(29,78,216,0.35);
        border-radius: 50%;
    }}
    .ag-ring-1 {{
        width: 60px; height: 60px;
        margin: -30px 0 0 -30px;
        border-color: rgba(29,78,216,0.50);
        animation: ag-spin-1 7s linear infinite;
    }}
    .ag-ring-2 {{
        width: 90px; height: 90px;
        margin: -45px 0 0 -45px;
        border-color: rgba(99,102,241,0.30);
        animation: ag-spin-2 12s linear infinite reverse;
    }}
    .ag-ring-3 {{
        width: 118px; height: 118px;
        margin: -59px 0 0 -59px;
        border-color: rgba(29,78,216,0.12);
        border-style: dashed;
        animation: ag-spin-1 22s linear infinite;
    }}
    .ag-ring-dot {{
        position: absolute;
        width: 6px; height: 6px;
        background: {ACCENT};
        border-radius: 50%;
        top: -3px; left: 50%;
        margin-left: -3px;
        box-shadow: 0 0 6px rgba(29,78,216,0.7);
    }}
    @keyframes ag-spin-1 {{
        from {{ transform: rotate(0deg);   }}
        to   {{ transform: rotate(360deg); }}
    }}
    @keyframes ag-spin-2 {{
        from {{ transform: rotate(0deg);   }}
        to   {{ transform: rotate(360deg); }}
    }}

    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────
def render_header():
    st.markdown(
        "<div class='ag-card ag-animate' style='padding:36px 40px;margin-bottom:28px;background:linear-gradient(135deg,#ffffff 0%,#eef2ff 100%);border-top:3px solid #1d4ed8;overflow:hidden;position:relative;'>"
        "<div style='position:absolute;top:-60px;right:-60px;width:280px;height:280px;border-radius:50%;background:radial-gradient(circle,rgba(29,78,216,0.06) 0%,transparent 65%);pointer-events:none;'></div>"
        "<div style='display:flex;align-items:center;gap:40px;position:relative;z-index:1;'>"
        "<div style='flex:1;'>"
        "<div style='display:inline-block;font-size:0.66rem;font-weight:700;letter-spacing:0.13em;text-transform:uppercase;color:#1d4ed8;background:#dbeafe;padding:4px 14px;border-radius:999px;margin-bottom:18px;'>AI Customer Intelligence</div>"
        "<div style='font-size:2.5rem;font-weight:800;color:#0f0f0f;letter-spacing:-0.04em;line-height:1.1;margin-bottom:12px;'>Churn Predictor</div>"
        "<div style='font-size:1rem;color:#4b5563;line-height:1.65;max-width:460px;'>Predictive intelligence for customer retention using XGBoost and CatBoost ensemble models.</div>"
        "</div>"
        "<div class='ag-orbital'>"
        "<div class='ag-ring ag-ring-3'><div class='ag-ring-dot'></div></div>"
        "<div class='ag-ring ag-ring-2'><div class='ag-ring-dot'></div></div>"
        "<div class='ag-ring ag-ring-1'><div class='ag-ring-dot'></div></div>"
        "<div class='ag-orbital-core'></div>"
        "</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  CLIENT FORM
# ─────────────────────────────────────────────────────────────
def render_client_form() -> dict:
    """Client input form — clean 3-column grid layout."""

    st.markdown("<div class='ag-group-label'>Demographics</div>", unsafe_allow_html=True)
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1:
        gender = st.selectbox("Gender", ['Male', 'Female'], label_visibility="visible")
    with r1c2:
        senior = st.selectbox("Senior Citizen", [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No")
    with r1c3:
        partner = st.selectbox("Partner", ['No', 'Yes'])
    with r1c4:
        dependents = st.selectbox("Dependents", ['No', 'Yes'])

    st.markdown("<div class='ag-group-label'>Subscription</div>", unsafe_allow_html=True)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
    with r2c2:
        contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    with r2c3:
        paperless = st.selectbox("Paperless Billing", ['No', 'Yes'])
    with r2c4:
        payment = st.selectbox(
            "Payment Method",
            ['Electronic check', 'Mailed check',
             'Bank transfer (automatic)', 'Credit card (automatic)']
        )

    st.markdown("<div class='ag-group-label'>Services</div>", unsafe_allow_html=True)
    r3c1, r3c2, r3c3 = st.columns(3)
    with r3c1:
        phone       = st.selectbox("Phone Service", ['Yes', 'No'])
        multi_lines = st.selectbox("Multiple Lines", ['No', 'Yes', 'No phone service'])
        internet    = st.selectbox("Internet Service", ['Fiber optic', 'DSL', 'No'])
    with r3c2:
        security    = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
        backup      = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
        device_prot = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    with r3c3:
        tech_support      = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
        streaming_tv      = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
        streaming_movies  = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])

    st.markdown("<div class='ag-group-label'>Financials</div>", unsafe_allow_html=True)
    fc1, fc2 = st.columns(2)
    with fc1:
        monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    with fc2:
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 10000.0,
            float(monthly_charges * max(tenure, 1)), step=10.0
        )

    return {
        'gender': gender, 'SeniorCitizen': senior,
        'Partner': partner, 'Dependents': dependents,
        'tenure': tenure, 'PhoneService': phone,
        'MultipleLines': multi_lines, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup,
        'DeviceProtection': device_prot, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    }


# ─────────────────────────────────────────────────────────────
#  PREDICTION RESULT
# ─────────────────────────────────────────────────────────────
def render_prediction_result(result: dict):
    is_churn   = result['prediction'] == 1
    accent_col = "#dc2626" if is_churn else "#16a34a"
    bg_col     = "rgba(220,38,38,0.05)" if is_churn else "rgba(22,163,74,0.05)"
    status_cls = "ag-status-churn" if is_churn else "ag-status-ok"
    label_text = result['label']
    risk_text  = result['risk_level'].replace("Risque", "Risk").replace(
        "Élevé", "High").replace("Modéré", "Moderate").replace("Faible", "Low")
    # Strip any emoji characters
    import re
    risk_text = re.sub(r'[^\x00-\x7F\u00C0-\u024F\u1E00-\u1EFF]+', '', risk_text).strip()

    st.markdown(
        f"<div class='ag-card {status_cls} ag-animate ag-delay-1' style='text-align:center;padding:28px 20px;background:{bg_col};'>"
        f"<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{accent_col};margin-bottom:8px;'>Prediction</div>"
        f"<div style='font-size:2.4rem;font-weight:800;color:{accent_col};letter-spacing:-0.03em;line-height:1;'>{label_text}</div>"
        f"<div style='margin-top:10px;font-size:0.9rem;font-weight:600;color:#374151;'>{risk_text}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Churn Probability", f"{result['probability_churn']:.1%}")
    with col2:
        st.metric("Retention Probability", f"{result['probability_no_churn']:.1%}")
    with col3:
        st.metric("Model", result['model_used'].upper())


# ─────────────────────────────────────────────────────────────
#  GAUGE CHART
# ─────────────────────────────────────────────────────────────
def render_gauge_chart(proba: float) -> go.Figure:
    pct = proba * 100
    bar_color = (
        "#dc2626" if pct >= 70 else
        "#f59e0b" if pct >= 40 else
        "#16a34a"
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Churn Risk Score", 'font': {'size': 15, 'color': '#374151', 'family': 'Inter'}},
        delta={'reference': 50,
               'increasing': {'color': "#dc2626"},
               'decreasing': {'color': "#16a34a"}},
        number={'font': {'size': 42, 'color': bar_color, 'family': 'Inter'},
                'suffix': ''},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#d1d5db",
                'tickfont': {'size': 11, 'color': '#6b7280'},
            },
            'bar': {'color': bar_color, 'thickness': 0.28},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0,  40],  'color': "rgba(22,163,74,0.08)"},
                {'range': [40, 70],  'color': "rgba(245,158,11,0.08)"},
                {'range': [70, 100], 'color': "rgba(220,38,38,0.08)"},
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 3},
                'thickness': 0.8,
                'value': pct,
            }
        }
    ))
    fig.update_layout(
        height=260,
        margin=dict(l=24, r=24, t=40, b=12),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Inter, sans-serif'},
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────
def render_confusion_matrix_plotly(model, X_test, y_test) -> go.Figure:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = ['No Churn', 'Churn']

    annotations = []
    for i in range(2):
        for j in range(2):
            val = int(cm[i, j])
            annotations.append(dict(
                x=labels[j], y=labels[i],
                text=str(val),
                showarrow=False,
                font=dict(size=22, color='white' if cm[i, j] > cm.max() * 0.5 else '#0f0f0f',
                          family='Inter')
            ))

    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, '#dbeafe'], [0.5, '#1d4ed8'], [1, '#0f0f0f']],
        showscale=False,
    ))
    fig.update_layout(
        title=dict(text='Confusion Matrix', font=dict(size=15, color='#0f0f0f', family='Inter')),
        xaxis_title='Predicted', yaxis_title='Actual',
        annotations=annotations,
        height=340, margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter'),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  ROC CURVE
# ─────────────────────────────────────────────────────────────
def render_roc_plotly(models: dict, X_test, y_test) -> go.Figure:
    palette = ['#1d4ed8', '#0f0f0f', '#6366f1', '#0891b2', '#7c3aed']
    fig = go.Figure()

    for i, (name, model) in enumerate(models.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"{name}  AUC {auc:.3f}",
            line=dict(color=palette[i % len(palette)], width=2.5),
            hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>',
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name='Random Baseline',
        line=dict(color='#9ca3af', width=1.2, dash='dot'),
    ))
    fig.update_layout(
        title=dict(text='ROC Curves', font=dict(size=15, color='#0f0f0f', family='Inter')),
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        legend=dict(x=0.55, y=0.12, bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='#e5e7eb', borderwidth=1, font=dict(size=11)),
        height=400, margin=dict(l=60, r=20, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter'),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        yaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────
def render_feature_importance_plotly(model, feature_names: list,
                                      model_name: str, top_n: int = 15) -> go.Figure:
    importances = model.feature_importances_
    fi_df = (pd.DataFrame({'feature': feature_names, 'importance': importances})
               .sort_values('importance', ascending=True)
               .tail(top_n))

    norm = fi_df['importance'] / fi_df['importance'].max()
    colors = [f"rgba(29,78,216,{0.25 + 0.75 * v:.2f})" for v in norm]

    fig = go.Figure(go.Bar(
        x=fi_df['importance'], y=fi_df['feature'],
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>',
    ))
    fig.update_layout(
        title=dict(text=f'Top {top_n} Features — {model_name}',
                   font=dict(size=15, color='#0f0f0f', family='Inter')),
        xaxis_title='Importance Score',
        height=460, margin=dict(l=160, r=40, t=50, b=60),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#374151', family='Inter'),
        xaxis=dict(gridcolor='rgba(0,0,0,0.05)', zeroline=False),
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  RECOMMENDATIONS
# ─────────────────────────────────────────────────────────────
def render_recommendations(recommendations: list):
    import re
    st.markdown("<div style='margin-top:6px;'><div class='ag-section-label'>Retention Recommendations</div></div>", unsafe_allow_html=True)

    for rec in recommendations:
        clean = re.sub(r'[^\x00-\x7F\u00C0-\u024F]+', '', rec).strip()
        clean = clean.lstrip(' -–—').strip()
        st.markdown(f"<div class='ag-rec-item'>{clean}</div>", unsafe_allow_html=True)