import base64
from pathlib import Path
from textwrap import dedent

import pandas as pd
import streamlit as st

from advisor import AdvisorError, OBJECTIVES, advisor_settings, generate_ai_advice
from valuation import (
    COMPARABLE_COLUMNS, DATASET_PATH, find_similar_apartments, get_statistics,
    load_dataset, predict_price, train_model,
)


# ==========================================
# Streamlit Configuration
# ==========================================

st.set_page_config(
    page_title="AI Apartment Price Predictor",
    page_icon="🏠",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent


# ==========================================
# Background and CSS Styling
# ==========================================

background_path = BASE_DIR / "architecture.svg"

if background_path.exists():
    background = base64.b64encode(
        background_path.read_bytes()
    ).decode()

    background_css = f"""
        linear-gradient(
            90deg,
            rgba(247,249,252,.13),
            rgba(247,249,252,.85) 32%,
            rgba(247,249,252,.85) 68%,
            rgba(247,249,252,.13)
        ),
        url('data:image/svg+xml;base64,{background}')
        center bottom / cover no-repeat fixed
    """
else:
    background_css = "#f7f9fc"


st.markdown(f"""
<style>

.stApp {{
    background: {background_css};
    color: #182c3b;
    color-scheme: light;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stMainBlockContainer"] {{
    max-width: 950px;
    padding-top: 3.5rem;
    padding-bottom: 3rem;
}}

[data-testid="stForm"] {{
    background: rgba(255,255,255,.96);
    border: 1px solid #dce5eb;
    border-radius: 20px;
    padding: 26px;
    box-shadow: 0 12px 36px rgba(29,53,72,.07);
}}

[data-testid="stMetric"] {{
    background: #ffffff;
    border: 1px solid #dce5eb;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 15px rgba(29,53,72,.04);
}}

[data-testid="stExpander"] {{
    background: rgba(255,255,255,.94);
    border-radius: 12px;
}}

[data-testid="stFormSubmitButton"] button {{
    background: #176b70;
    color: white;
    border: none;
    border-radius: 10px;
    min-height: 48px;
    font-weight: 600;
}}

[data-testid="stFormSubmitButton"] button:hover {{
    background: #12565a;
    color: white;
}}

.property-hero {{
    background: rgba(255,255,255,.95);
    border: 1px solid #dce5eb;
    border-left: 5px solid #176b70;
    border-radius: 18px;
    padding: 26px 30px;
    margin-bottom: 20px;
}}

.property-hero h1 {{
    font-size: clamp(1.65rem, 4vw, 2.35rem);
    margin: 8px 0;
    padding: 0;
    color: #182c3b;
}}

.property-hero p {{
    color: #526877;
    margin: 0;
}}

.property-kicker {{
    color: #176b70;
    font-size: .76rem;
    letter-spacing: .15em;
    font-weight: 700;
}}

.ai-header {{
    background: linear-gradient(135deg, #176b70, #123e52);
    border-radius: 16px;
    padding: 22px;
    margin-top: 20px;
    margin-bottom: 20px;
    color: white;
}}

.ai-header h2 {{
    color: white;
    margin: 0;
    padding: 0;
}}

.ai-header p {{
    color: #e1eff1;
    margin-top: 8px;
    margin-bottom: 0;
}}

</style>
""", unsafe_allow_html=True)


# ==========================================
# Train Machine Learning Model
# ==========================================

@st.cache_resource
def load_app_data(dataset_version):
    # The modified timestamp participates in the cache key.
    data = load_dataset()
    model, metrics = train_model(data)
    return data, model, metrics


try:
    df, model, metrics = load_app_data(DATASET_PATH.stat().st_mtime_ns)
except (OSError, ValueError):
    st.error("Could not load the apartment dataset. Check apartment_prices.csv and its required fields.")
    st.stop()


# ==========================================
# Page Header
# ==========================================

st.markdown("""
<style>
.property-hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(
        115deg, #ffffff 50%, #eaf3f4 100%
    );
    border: 1px solid #dce5eb;
    border-left: 6px solid #176b70;
    border-radius: 22px;
    padding: 35px;
    margin-bottom: 25px;
    box-shadow: 0 10px 35px rgba(29, 53, 72, 0.06);
}

.hero-content {
    position: relative;
    z-index: 2;
}

.property-kicker {
    color: #176b70;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 700;
    margin-bottom: 18px;
}

.property-hero h1 {
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -1.5px;
    color: #182c3b;
    margin: 0;
    padding: 0;
}

.hero-highlight {
    color: #176b70;
}

.property-hero p {
    color: #647787;
    font-size: 15px;
    line-height: 1.8;
    margin-top: 16px;
    max-width: 480px;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}

.hero-tag {
    background: #e8f3f2;
    border: 1px solid #d4e8e6;
    color: #176b70;
    font-size: 11px;
    font-weight: 600;
    padding: 8px 13px;
    border-radius: 7px;
}

.hero-building {
    position: absolute;
    right: 15px;
    bottom: -10px;
    width: 260px;
    opacity: 0.14;
    pointer-events: none;
}

@media (max-width: 600px) {
    .property-hero {
        padding: 24px;
    }

    .hero-building {
        width: 160px;
        opacity: 0.07;
    }
}
</style>
""", unsafe_allow_html=True)


hero_html = """
<div class="property-hero">
<div class="hero-content">
<div class="property-kicker">● REAL ESTATE INTELLIGENCE</div>
<h1>AI Apartment<br><span class="hero-highlight">Price Predictor</span></h1>
<p>Smarter property decisions powered by data and artificial intelligence.</p>
<div class="hero-tags">
<span class="hero-tag">Machine Learning</span>
<span class="hero-tag">Market Analytics</span>
<span class="hero-tag">AI Recommendations</span>
</div>
</div>
<svg class="hero-building" viewBox="0 0 240 260" xmlns="http://www.w3.org/2000/svg" fill="none" stroke="#176b70" stroke-width="1.5">
<path d="M45 250V65L120 15L195 65V250"/>
<path d="M120 15V250"/>
<path d="M45 65L120 115L195 65"/>
<path d="M15 250V145L45 125"/>
<path d="M195 125L225 145V250"/>
<path d="M0 250H240"/>
<rect x="60" y="120" width="18" height="22"/>
<rect x="88" y="120" width="18" height="22"/>
<rect x="135" y="120" width="18" height="22"/>
<rect x="163" y="120" width="18" height="22"/>
<rect x="60" y="155" width="18" height="22"/>
<rect x="88" y="155" width="18" height="22"/>
<rect x="135" y="155" width="18" height="22"/>
<rect x="163" y="155" width="18" height="22"/>
<rect x="60" y="190" width="18" height="22"/>
<rect x="88" y="190" width="18" height="22"/>
<rect x="135" y="190" width="18" height="22"/>
<rect x="163" y="190" width="18" height="22"/>
</svg>
</div>
"""

st.html(hero_html)

# ==========================================
# User Input Form
# ==========================================

with st.form("prediction_form"):

    st.subheader("🏠 Apartment Details")

    city = st.selectbox(
        "City",
        sorted(df["city"].unique())
    )

    left, right = st.columns(2)

    with left:

        area = st.number_input(
            "Area (m²)",
            min_value=75,
            max_value=219,
            value=140,
            step=1
        )

        rooms = st.number_input(
            "Number of Rooms",
            min_value=2,
            max_value=6,
            value=3,
            step=1
        )

    with right:

        floor = st.number_input(
            "Floor",
            min_value=1,
            max_value=10,
            value=3,
            step=1
        )

        asking_price = st.number_input(
            "Asking Price ($) — Optional",
            min_value=0,
            value=0,
            step=1000
        )

    st.subheader("🎯 User Preferences")

    purpose = st.selectbox(
        "What is your objective?",
        OBJECTIVES
    )

    language = st.selectbox(
        "AI Analysis Language",
        ["Arabic", "English"]
    )

    enable_ai = st.checkbox(
        "Generate AI Recommendations",
        value=False,
        help="Optional. Sends property inputs and aggregate dataset statistics to OpenAI."
    )

    submitted = st.form_submit_button(
        "Analyze Property",
        type="primary",
        width="stretch"
    )


# ==========================================
# Prediction
# ==========================================

if submitted:

    prediction = predict_price(model, city, area, rooms, floor)
    city_stats = get_statistics(df.loc[df["city"].eq(city)])
    dataset_stats = get_statistics(df)
    city_average = city_stats["average_price_usd"]
    dataset_average = dataset_stats["average_price_usd"]
    price_per_sqm = prediction / area
    similar = find_similar_apartments(df, city, area, rooms)
    similar_stats = get_statistics(similar)
    asking_delta = (asking_price - prediction) / prediction * 100 if asking_price else None

    st.success("Property prediction completed successfully!")


    # ======================================
    # Property Valuation
    # ======================================

    st.subheader("💰 Property Valuation")

    col1, col2 = st.columns(2)

    col1.metric(
        "Estimated Apartment Price",
        f"${prediction:,.2f}"
    )

    col2.metric(
        "Price per Square Meter",
        f"${price_per_sqm:,.2f}"
    )

    col3, col4 = st.columns(2)

    col3.metric(
        "City Average Price",
        f"${city_average:,.2f}"
    )

    col4.metric(
        "Dataset Average Price",
        f"${dataset_average:,.2f}"
    )

    if asking_delta is not None:
        st.metric(
            "Asking Price",
            f"${asking_price:,.2f}",
            delta=f"{asking_delta:+.2f}% vs prediction",
            delta_color="off"
        )

    st.caption(
        "The estimated price is generated by the "
        "machine learning model."
    )

    st.divider()


    # ======================================
    # Market Comparison
    # ======================================

    st.subheader("📊 Property Price Comparison")

    comparison_data = {

        "Predicted Price": prediction,

        "City Average": city_average,

        "Dataset Average": dataset_average
    }

    if not similar.empty:

        comparison_data["Similar Apartments"] = similar_stats["average_price_usd"]

    if asking_price > 0:

        comparison_data["Asking Price"] = asking_price

    comparison = pd.DataFrame(
        list(comparison_data.items()),
        columns=["Category", "Price (USD)"]
    )

    st.bar_chart(
        comparison.set_index("Category"),
        color="#176b70"
    )

    st.divider()


    # ======================================
    # Similar Apartments
    # ======================================

    st.subheader("🏢 Similar Apartments")

    if not similar.empty:

        st.write(
            f"Found **{similar_stats['count']}** comparable apartments "
            f"in {city}."
        )

        a, b, c = st.columns(3)

        a.metric(
            "Average Price",
            f"${similar_stats['average_price_usd']:,.2f}"
        )

        b.metric(
            "Lowest Price",
            f"${similar_stats['minimum_price_usd']:,.2f}"
        )

        c.metric(
            "Highest Price",
            f"${similar_stats['maximum_price_usd']:,.2f}"
        )

        st.dataframe(
            similar[COMPARABLE_COLUMNS],
            width="stretch",
            hide_index=True
        )

    else:

        st.warning(
            "No closely matching apartments were found "
            "in the dataset."
        )

    st.divider()


    with st.expander("Model evaluation (held-out dataset)"):
        st.write(
            f"R²: {metrics['r2']:.4f} | MAE: USD {metrics['mae']:,.2f} | "
            f"RMSE: USD {metrics['rmse']:,.2f} | Test rows: {metrics['test_rows']}"
        )
        st.caption(
            "R² measures fit on the test set. MAE and RMSE summarize test-set errors. "
            "These metrics do not quantify uncertainty for an individual apartment."
        )

    if enable_ai:
        st.html(dedent("""
        <div class="ai-header">
            <h2>✨ AI Property Advisor</h2>
            <p>Recommendations based on the estimate, dataset statistics, and your objective.</p>
        </div>
        """))

        api_key, ai_model = advisor_settings()
        if not api_key:
            st.warning("Set OPENAI_API_KEY in your environment or local .env to enable the advisor.")
        else:
            context = {
                "city": city,
                "area_sqm": int(area),
                "rooms": int(rooms),
                "floor": int(floor),
                "estimated_price_usd": round(prediction, 2),
                "estimated_price_per_sqm_usd": round(price_per_sqm, 2),
                "asking_price_usd": float(asking_price) if asking_price else None,
                "asking_vs_estimate_percent": round(asking_delta, 2) if asking_delta is not None else None,
                "city_statistics": city_stats,
                "dataset_statistics": dataset_stats,
                "comparable_statistics": similar_stats,
                "comparable_criteria": "Same city and room count, area within 20% of the requested area.",
            }
            with st.spinner("AI is selecting a recommendation…"):
                try:
                    sections = generate_ai_advice(api_key, ai_model, context, purpose, language)
                except AdvisorError as error:
                    st.warning(str(error))
                else:
                    for heading, body in sections:
                        st.markdown(f"**{heading}**")
                        st.markdown(body.replace("$", r"\$"))


# ==========================================
# Footer
# ==========================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#526877;
    font-size:13px;
    padding:15px;
">
    AI Real Estate Decision Support System<br>
    Random Forest × OpenAI × Streamlit<br><br>
    Estimates and AI recommendations are for
    informational purposes and are not certified
    property valuations.
</div>
""", unsafe_allow_html=True)
