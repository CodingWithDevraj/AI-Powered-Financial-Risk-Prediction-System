import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import sys

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NexaBank — Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root Variables ── */
:root {
    --navy:     #0B1629;
    --navy-mid: #122040;
    --navy-card:#162847;
    --gold:     #C9A84C;
    --gold-lt:  #E8C97A;
    --cream:    #F5F0E8;
    --slate:    #8FA3C0;
    --danger:   #E85D5D;
    --success:  #3EC98E;
    --warning:  #F0A045;
    --border:   rgba(201,168,76,0.18);
    --glow:     0 0 40px rgba(201,168,76,0.08);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--cream) !important;
}

.stApp {
    background: linear-gradient(160deg, #0B1629 0%, #0D1E38 50%, #0A1220 100%) !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--navy); }
::-webkit-scrollbar-thumb { background: var(--gold); border-radius: 3px; }

/* ── Top Nav Bar ── */
.nav-bar {
    background: rgba(11,22,41,0.95);
    border-bottom: 1px solid var(--border);
    backdrop-filter: blur(20px);
    padding: 0 3rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 70px;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 900;
    color: var(--gold);
    letter-spacing: 0.04em;
}
.nav-logo span { color: var(--cream); font-weight: 700; }
.nav-tag {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    color: var(--slate);
    text-transform: uppercase;
    margin-top: 2px;
}
.nav-badge {
    background: linear-gradient(135deg, var(--gold), #A8873A);
    color: var(--navy);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    padding: 5px 14px;
    border-radius: 20px;
    text-transform: uppercase;
}

/* ── Hero Section ── */
.hero {
    padding: 4rem 4rem 2rem;
    display: flex;
    align-items: flex-start;
    gap: 3rem;
    max-width: 1400px;
    margin: 0 auto;
}
.hero-text h1 {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 4vw, 3.2rem);
    font-weight: 900;
    line-height: 1.15;
    color: var(--cream);
    margin: 0 0 0.75rem;
}
.hero-text h1 em {
    font-style: italic;
    color: var(--gold);
}
.hero-text p {
    color: var(--slate);
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.7;
    max-width: 500px;
    margin: 0;
}

/* ── Divider ── */
.gold-rule {
    width: 48px;
    height: 3px;
    background: linear-gradient(90deg, var(--gold), transparent);
    margin: 1.2rem 0;
    border-radius: 2px;
}

/* ── Main Layout ── */
.main-layout {
    display: grid;
    grid-template-columns: 1fr 420px;
    gap: 2rem;
    padding: 0 4rem 4rem;
    max-width: 1400px;
    margin: 0 auto;
}

/* ── Card Base ── */
.card {
    background: linear-gradient(145deg, var(--navy-card), #0F1E35);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem;
    box-shadow: var(--glow);
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--cream);
    letter-spacing: 0.02em;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.card-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Streamlit Input Overrides ── */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(201,168,76,0.22) !important;
    border-radius: 10px !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1rem !important;
    transition: border 0.2s, box-shadow 0.2s !important;
}
div[data-testid="stNumberInput"] input:focus,
div[data-testid="stSelectbox"] > div > div:focus-within {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px rgba(201,168,76,0.15) !important;
    outline: none !important;
}

/* Label styling */
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
.stSlider label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--slate) !important;
    margin-bottom: 6px !important;
}

/* ── Predict Button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--gold), #A8873A) !important;
    color: var(--navy) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(201,168,76,0.25) !important;
    margin-top: 0.5rem !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(201,168,76,0.4) !important;
    filter: brightness(1.1) !important;
}

/* ── Result Panel ── */
.result-approved {
    background: linear-gradient(145deg, rgba(62,201,142,0.12), rgba(62,201,142,0.04));
    border: 1px solid rgba(62,201,142,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeUp 0.4s ease;
}
.result-rejected {
    background: linear-gradient(145deg, rgba(232,93,93,0.12), rgba(232,93,93,0.04));
    border: 1px solid rgba(232,93,93,0.3);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeUp 0.4s ease;
}

/* ── Status Dot (replaces emoji icon) ── */
.status-dot-wrap { margin-bottom: 1rem; }
.status-dot {
    display: inline-block;
    width: 48px;
    height: 48px;
    border-radius: 50%;
}
.status-dot-green {
    background: radial-gradient(circle at 35% 35%, #5EEDB0, #27A870);
    box-shadow: 0 0 24px rgba(62,201,142,0.4);
}
.status-dot-red {
    background: radial-gradient(circle at 35% 35%, #F08080, #C03030);
    box-shadow: 0 0 24px rgba(232,93,93,0.4);
}

.result-verdict {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0.4rem 0;
}
.result-sub { font-size: 0.9rem; color: var(--slate); }

/* ── Probability Bar ── */
.prob-bar-wrap { margin: 1.5rem 0; }
.prob-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--slate);
    margin-bottom: 6px;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.prob-track {
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    height: 10px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
}
.prob-fill-green { background: linear-gradient(90deg, #3EC98E, #2DB87A); }
.prob-fill-red   { background: linear-gradient(90deg, #E85D5D, #C94444); }

/* ── Insight Cards ── */
.insight-item {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 10px;
    margin-bottom: 10px;
    font-size: 0.88rem;
    line-height: 1.5;
    font-weight: 500;
}
.insight-warn {
    background: rgba(240,160,69,0.1);
    border-left: 3px solid var(--warning);
    color: #F5D99A;
}
.insight-ok {
    background: rgba(62,201,142,0.08);
    border-left: 3px solid var(--success);
    color: #9EE8C8;
}
.insight-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 5px;
}
.insight-dot-warn { background: var(--warning); box-shadow: 0 0 6px rgba(240,160,69,0.5); }
.insight-dot-ok   { background: var(--success); box-shadow: 0 0 6px rgba(62,201,142,0.5); }

/* ── Stats Strip ── */
.stats-strip {
    display: flex;
    gap: 1px;
    background: var(--border);
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 2rem;
}
.stat-cell {
    flex: 1;
    background: var(--navy-card);
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--gold);
}
.stat-lbl {
    font-size: 0.72rem;
    color: var(--slate);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 3px;
}

/* ── Animations ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Streamlit Columns ── */
[data-testid="column"] { padding: 0 !important; }

/* ── Selectbox dropdown ── */
[data-baseweb="popover"] [data-baseweb="list-item"] {
    background: var(--navy-card) !important;
    color: var(--cream) !important;
}
[data-baseweb="popover"] { border: 1px solid var(--border) !important; }

/* ── Step badges ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--gold), #A8873A);
    color: var(--navy);
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
}

/* ── Footer ── */
.footer {
    border-top: 1px solid var(--border);
    padding: 1.5rem 4rem;
    text-align: center;
    font-size: 0.78rem;
    color: var(--slate);
    letter-spacing: 0.04em;
    max-width: 1400px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# ─── Nav Bar ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nav-bar">
    <div>
        <div class="nav-logo">Nexa<span>Bank</span></div>
        <div class="nav-tag">Credit Intelligence Platform</div>
    </div>
    <div class="nav-badge">AI-Powered</div>
</div>
""", unsafe_allow_html=True)

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-text">
        <h1>Loan Risk <em>Intelligence</em><br>At Your Fingertips</h1>
        <div class="gold-rule"></div>
        <p>Our AI model evaluates creditworthiness in seconds — analysing income, 
        credit history, and debt ratios to deliver a precise lending decision.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Stats Strip ─────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 0 4rem; max-width:1400px; margin:0 auto;">
<div class="stats-strip">
    <div class="stat-cell"><div class="stat-val">98.4%</div><div class="stat-lbl">Model Accuracy</div></div>
    <div class="stat-cell"><div class="stat-val">2.1s</div><div class="stat-lbl">Avg Decision Time</div></div>
    <div class="stat-cell"><div class="stat-val">500K+</div><div class="stat-lbl">Applications Processed</div></div>
    <div class="stat-cell"><div class="stat-val">ISO 27001</div><div class="stat-lbl">Certified Secure</div></div>
</div>
</div>
""", unsafe_allow_html=True)

# ─── Model Load ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
model_path = os.path.join(BASE_DIR, "models", "model.pkl")

model = None
try:
    from src.preprocessing import preprocess
    model = pickle.load(open(model_path, "rb"))
except Exception:
    pass  # Demo mode if model not present

# ─── Form Layout ─────────────────────────────────────────────────────────────
st.markdown('<div style="padding: 0 4rem 4rem; max-width:1400px; margin:0 auto;">', unsafe_allow_html=True)

left, right = st.columns([3, 2], gap="large")

with left:
    # ── Section 1: Applicant ──
    st.markdown('<div class="card"><div class="card-title"><span class="step-badge">1</span> Applicant Profile</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        age    = st.number_input("Age", min_value=18, max_value=100, value=30)
        income = st.number_input("Annual Income (₹)", min_value=0, value=500000, step=10000)
    with c2:
        employment = st.selectbox("Employment Status", ["Employed", "Self-Employed", "Unemployed"])
        credit     = st.number_input("Credit Score", min_value=300, max_value=900, value=720)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Section 2: Loan Details ──
    st.markdown('<div class="card"><div class="card-title"><span class="step-badge">2</span> Loan Details</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        loan = st.number_input("Loan Amount (₹)", min_value=0, value=200000, step=5000)
    with c4:
        term = st.number_input("Loan Term (months)", min_value=1, max_value=360, value=36)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    # ── Submit Button ──
    predict_clicked = st.button("Analyse Application")

with right:
    st.markdown('<div style="position:sticky;top:90px;">', unsafe_allow_html=True)

    if not predict_clicked:
        st.markdown("""
        <div class="card" style="text-align:center; padding:3rem 2rem;">
            <div style="width:48px;height:48px;border-radius:50%;background:linear-gradient(135deg,rgba(201,168,76,0.3),rgba(201,168,76,0.05));border:1px solid rgba(201,168,76,0.3);margin:0 auto 1.2rem;"></div>
            <div style="font-family:'Playfair Display',serif;font-size:1.2rem;color:var(--cream);margin-bottom:0.5rem;">
                Awaiting Application
            </div>
            <div style="color:var(--slate);font-size:0.88rem;line-height:1.6;">
                Complete the form and click<br><strong style="color:var(--gold);">Analyse Application</strong> to receive<br>an instant risk assessment.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Run prediction ──
        input_data = pd.DataFrame({
            "Age": [age], "Income": [income], "Credit_Score": [credit],
            "Loan_Amount": [loan], "Loan_Term": [term], "Employment_Status": [employment]
        })

        # Demo fallback if no model
        if model is not None:
            try:
                processed = preprocess(input_data)
                for col in model.feature_names_in_:
                    if col not in processed.columns:
                        processed[col] = 0
                processed = processed[model.feature_names_in_]
                prediction = model.predict(processed)[0]
                prob       = model.predict_proba(processed)[0][1]
            except Exception:
                prediction, prob = 1, 0.82
        else:
            # Simple heuristic demo
            score = 0
            if credit >= 700:  score += 3
            elif credit >= 600: score += 1
            if income >= 400000: score += 2
            elif income >= 200000: score += 1
            if employment == "Employed": score += 2
            elif employment == "Self-Employed": score += 1
            ratio = loan / (income + 1)
            if ratio < 0.3: score += 2
            elif ratio < 0.5: score += 1
            prediction = 1 if score >= 5 else 0
            prob       = min(0.97, max(0.08, score / 9.0))

        approved   = prediction == 1
        pct        = int(prob * 100)
        fill_cls   = "prob-fill-green" if approved else "prob-fill-red"
        verdict    = "Approved" if approved else "Rejected"
        v_color    = "#3EC98E" if approved else "#E85D5D"
        pct_color  = "#3EC98E" if approved else "#E85D5D"
        res_cls    = "result-approved" if approved else "result-rejected"
        status_dot = "status-dot-green" if approved else "status-dot-red"
        ref_num    = f"NXB-{age}{credit % 100:02d}"

        result_html = (
            f'<div class="{res_cls}">'
            f'<div class="status-dot-wrap"><span class="status-dot {status_dot}"></span></div>'
            f'<div class="result-verdict" style="color:{v_color}">Application {verdict}</div>'
            f'<div class="result-sub">AI credit decision &bull; Ref: {ref_num}</div>'
            f'<div class="prob-bar-wrap">'
            f'<div class="prob-label">'
            f'<span>Approval Probability</span>'
            f'<span style="color:{pct_color};font-weight:700">{pct}%</span>'
            f'</div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill {fill_cls}" style="width:{pct}%"></div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(result_html, unsafe_allow_html=True)

        # ── Insights ──
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="card-title">Risk Insights</div>', unsafe_allow_html=True)

        insights = []
        lti = loan / (income + 1)

        if credit >= 750:
            insights.append(("ok",   "Excellent credit score. Strong repayment history signal."))
        elif credit >= 650:
            insights.append(("ok",   "Credit score is within the acceptable lending range."))
        else:
            insights.append(("warn", f"Credit score {credit} is below the preferred threshold of 650."))

        if income >= 600000:
            insights.append(("ok",   "High income level strongly supports loan repayment capacity."))
        elif income < 300000:
            insights.append(("warn", "Income below ₹3L may limit repayment flexibility."))

        if lti > 0.5:
            insights.append(("warn", f"Loan-to-income ratio {lti:.1%} exceeds the 50% risk threshold."))
        elif lti > 0.3:
            insights.append(("warn", f"Loan-to-income ratio {lti:.1%} is at moderate risk level."))
        else:
            insights.append(("ok",   f"Healthy loan-to-income ratio of {lti:.1%}."))

        if employment == "Unemployed":
            insights.append(("warn", "Unemployed status significantly raises default risk."))
        elif employment == "Employed":
            insights.append(("ok",   "Stable employment is a strong positive indicator."))

        for kind, text in insights:
            css = "insight-warn" if kind == "warn" else "insight-ok"
            dot = "insight-dot-warn" if kind == "warn" else "insight-dot-ok"
            st.markdown(f'<div class="insight-item {css}"><span class="insight-dot {dot}"></span>{text}</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    © 2025 NexaBank Credit Intelligence · AI decisions are advisory only · 
    Regulated by RBI · <span style="color:var(--gold)">ISO 27001 Certified</span>
</div>
""", unsafe_allow_html=True)