import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

from ai_risk_pricing.config import DEFAULT_CONFIG, ModelConfig, PricingParams
from ai_risk_pricing.portfolio import Portfolio, PortfolioAggregator
from ai_risk_pricing.portfolio.company import Company, Sector
from ai_risk_pricing.scenario import ScenarioGenerator
from ai_risk_pricing.modeling import MonteCarloEngine, DependencyGraph
from ai_risk_pricing.pricing import (
    RiskMetrics,
    PremiumCalculator,
    IndividualPremiumCalculator,
    estimate_safety_investment_benefit,
)
from ai_risk_pricing.safety import (
    RiskSurface,
    SafetyProfile,
    MitigationEngine,
    get_registry,
    build_profile_from_selections,
)

# =============================================================================
# PAGE CONFIG & STYLING
# =============================================================================

st.set_page_config(
    page_title="AI Risk Pricing Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark actuarial theme - matching landing page gold/black palette
DARK_THEME = """
<style>
    /* Import Inter and JetBrains Mono fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
    
    /* Main background - deep black */
    .stApp {
        background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0a 0%, #111111 100%);
        border-right: 1px solid #2a2a2a;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f5f5f5 !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: -0.02em;
    }
    
    h1 {
        background: linear-gradient(135deg, #c9a962 0%, #9a7d3e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        color: #c9a962 !important;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.8rem !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #a0a0a0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Cards/containers */
    .stAlert {
        background-color: rgba(15, 15, 15, 0.9) !important;
        border: 1px solid #2a2a2a !important;
    }
    
    /* Buttons - gold gradient */
    .stButton > button {
        background: linear-gradient(135deg, #c9a962 0%, #9a7d3e 100%);
        color: #0a0a0a;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(201, 169, 98, 0.3);
    }
    
    /* Sliders */
    .stSlider > div > div {
        background-color: #2a2a2a !important;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #111111 !important;
        border-color: #2a2a2a !important;
    }
    
    /* Input fields */
    .stNumberInput > div > div > input {
        background-color: #111111 !important;
        border-color: #2a2a2a !important;
        color: #f5f5f5 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(15, 15, 15, 0.8) !important;
        border-radius: 8px;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid #2a2a2a !important;
        border-radius: 8px;
    }
    
    /* Custom metric card */
    .metric-card {
        background: #0f0f0f;
        border: 1px solid #2a2a2a;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(135deg, #c9a962 0%, #9a7d3e 100%);
        opacity: 0.5;
    }
    
    .metric-card h4 {
        color: #a0a0a0 !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.5rem;
    }
    
    .metric-card .value {
        color: #c9a962;
        font-size: 2rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
    }
    
    /* Divider line */
    hr {
        border-color: #2a2a2a !important;
    }
    
    /* Risk indicator colors */
    .risk-low { color: #22c55e !important; }
    .risk-medium { color: #f59e0b !important; }
    .risk-high { color: #ef4444 !important; }
    
    /* Actuarial accent */
    .actuarial-accent {
        border-left: 3px solid #c9a962;
        padding-left: 1rem;
        margin: 1rem 0;
    }
    
    /* Premium highlight */
    .premium-highlight {
        background: rgba(201, 169, 98, 0.05);
        border: 1px solid rgba(201, 169, 98, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #0f0f0f;
        border-radius: 8px 8px 0 0;
        border: 1px solid #2a2a2a;
        border-bottom: none;
        color: #a0a0a0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(201, 169, 98, 0.1);
        color: #c9a962;
        border-color: #9a7d3e;
    }
</style>
"""

st.markdown(DARK_THEME, unsafe_allow_html=True)

# =============================================================================
# PLOTLY DARK THEME
# =============================================================================

PLOTLY_TEMPLATE = {
    "layout": {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(15,15,15,0.8)",
        "font": {"color": "#f5f5f5", "family": "Inter, -apple-system, sans-serif"},
        "title": {"font": {"color": "#f5f5f5", "size": 16}},
        "xaxis": {
            "gridcolor": "rgba(42,42,42,0.4)",
            "linecolor": "#2a2a2a",
            "tickfont": {"color": "#a0a0a0"},
        },
        "yaxis": {
            "gridcolor": "rgba(42,42,42,0.4)",
            "linecolor": "#2a2a2a",
            "tickfont": {"color": "#a0a0a0"},
        },
        "colorway": ["#c9a962", "#e4c87a", "#22c55e", "#f59e0b", "#ef4444", "#9a7d3e"],
        "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"color": "#a0a0a0"}},
    }
}


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """Apply dark actuarial theme to Plotly figure matching landing page style."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,15,15,0.6)",
        font=dict(color="#f5f5f5", family="Inter, -apple-system, sans-serif"),
        xaxis=dict(gridcolor="rgba(42,42,42,0.4)", linecolor="#2a2a2a"),
        yaxis=dict(gridcolor="rgba(42,42,42,0.4)", linecolor="#2a2a2a"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#a0a0a0")),
    )
    return fig


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data(ttl=3600)
def run_portfolio_simulation(
    n_companies: int,
    n_years: int,
    include_dark: bool,
    seed: int,
) -> tuple:
    """Run full portfolio simulation (cached for performance)."""
    
    # Build portfolio
    portfolio = Portfolio.build_sample_portfolio(n_companies=n_companies, seed=seed)
    
    # Generate scenarios
    generator = ScenarioGenerator(rng=np.random.default_rng(seed))
    scenarios = generator.get_all_scenarios(include_dark=include_dark)
    
    # Build dependency graph
    aggregator = PortfolioAggregator(portfolio)
    graph = aggregator.build_dependency_graph_from_portfolio(
        n_foundation_models=2,
        n_saas_providers=4,
    )
    
    # Run simulation
    config = DEFAULT_CONFIG
    engine = MonteCarloEngine(
        scenarios=scenarios,
        dependency_graph=graph,
        config=config,
        seed=seed,
    )
    
    result = engine.simulate_years_vectorized(n_years=n_years)
    
    # Calculate metrics
    metrics = RiskMetrics(result.year_loss_table)
    risk_results = metrics.calculate_all(portfolio.total_exposure)
    
    # Calculate premium
    calculator = PremiumCalculator(
        risk_metrics=metrics,
        total_exposure=portfolio.total_exposure,
        params=config.pricing,
    )
    premium_breakdown = calculator.calculate_premium()
    
    return (
        portfolio,
        scenarios,
        graph,
        result,
        risk_results,
        premium_breakdown,
        metrics,
    )


def create_metric_card(label: str, value: str, delta: str = None, delta_color: str = "normal") -> str:
    """Create a styled metric card HTML."""
    delta_html = ""
    if delta:
        color = "#00d4aa" if delta_color == "normal" else "#ff6b6b"
        delta_html = f'<div style="color: {color}; font-size: 0.9rem;">{delta}</div>'
    
    return f"""
    <div class="metric-card">
        <h4>{label}</h4>
        <div class="value">{value}</div>
        {delta_html}
    </div>
    """


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚡ AI Risk Engine")
    st.markdown("---")
    
    # Navigation
    page = st.radio(
        "Navigation",
        ["📊 Portfolio Metrics", "🏢 New Company Simulation"],
        label_visibility="collapsed",
    )
    
    st.markdown("---")
    st.markdown("### Simulation Parameters")
    
    n_companies = st.slider(
        "Portfolio Companies",
        min_value=5,
        max_value=50,
        value=15,
        help="Number of companies in the portfolio",
    )
    
    n_years = st.select_slider(
        "Simulation Years",
        options=[1000, 5000, 10000, 25000, 50000, 100000],
        value=10000,
        help="More years = more accurate tail estimates",
    )
    
    include_dark = st.toggle(
        "Include Dark Scenario",
        value=False,
        help="Inject extreme tail events for stress testing",
    )
    
    seed = st.number_input(
        "Random Seed",
        min_value=1,
        max_value=9999,
        value=42,
        help="For reproducibility",
    )
    
    st.markdown("---")
    
    # Pricing Parameters
    with st.expander("⚙️ Pricing Parameters"):
        ambiguity_load = st.slider(
            "Ambiguity Load (α)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Loading factor for parameter uncertainty",
        )
        
        expense_ratio = st.slider(
            "Expense Ratio (ε)",
            min_value=0.1,
            max_value=0.5,
            value=0.25,
            step=0.05,
            help="Operating costs as % of EL",
        )
    
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666666; font-size: 0.75rem;">
            <span style="color: #c9a962;">AI Catastrophe Model</span> v0.1<br>
            Quantifying the Unquantifiable
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# PAGE 1: PORTFOLIO METRICS
# =============================================================================

if page == "📊 Portfolio Metrics":
    st.markdown("# Portfolio Risk Analytics")
    st.markdown(
        '<p style="color: #a0a0a0; font-size: 1.1rem;">Comprehensive view of AI catastrophe risk exposure and premium calculation</p>',
        unsafe_allow_html=True,
    )
    
    # Run simulation
    with st.spinner("Running Monte Carlo simulation..."):
        (
            portfolio,
            scenarios,
            graph,
            sim_result,
            risk_results,
            premium_breakdown,
            risk_metrics,
        ) = run_portfolio_simulation(n_companies, n_years, include_dark, seed)
    
    # Top-level metrics
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Exposure",
            f"${portfolio.total_exposure:,.0f}M",
            delta=f"{n_companies} companies",
        )
    
    with col2:
        st.metric(
            "Expected Loss (EL)",
            f"${risk_results.expected_loss:,.2f}M",
            delta=f"{risk_results.loss_ratio:.2%} loss ratio",
        )
    
    with col3:
        st.metric(
            "TVaR 99%",
            f"${risk_results.tvar_99:,.2f}M",
            delta=f"{risk_results.tvar_99 / risk_results.expected_loss:.1f}x EL" if risk_results.expected_loss > 0 else "N/A",
        )
    
    with col4:
        st.metric(
            "Technical Premium",
            f"${premium_breakdown.total_premium:,.2f}M",
            delta=f"{premium_breakdown.rate_on_line:.2f}% RoL",
        )
    
    # Premium breakdown and exceedance curve
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1.5])
    
    with col_left:
        st.markdown("### Premium Breakdown")
        
        # Premium waterfall chart
        premium_data = {
            "Component": ["Expected Loss", "Ambiguity Load", "Expense Load", "Total Premium"],
            "Amount": [
                premium_breakdown.expected_loss,
                premium_breakdown.ambiguity_load,
                premium_breakdown.expense_load,
                premium_breakdown.total_premium,
            ],
        }
        
        fig_premium = go.Figure()
        
        # Waterfall-style bars - gold palette
        colors = ["#c9a962", "#e4c87a", "#f59e0b", "#22c55e"]
        
        fig_premium.add_trace(go.Bar(
            x=premium_data["Component"],
            y=premium_data["Amount"],
            marker_color=colors,
            text=[f"${v:,.2f}M" for v in premium_data["Amount"]],
            textposition="outside",
            textfont=dict(color="#f5f5f5"),
        ))
        
        fig_premium = apply_dark_theme(fig_premium)
        fig_premium.update_layout(
            height=350,
            showlegend=False,
            yaxis_title="Premium ($M)",
            margin=dict(t=20, b=20),
        )
        
        st.plotly_chart(fig_premium, use_container_width=True)
        
        # Key ratios
        st.markdown("#### Key Ratios")
        ratio_col1, ratio_col2 = st.columns(2)
        with ratio_col1:
            st.metric("Rate on Line", f"{premium_breakdown.rate_on_line:.2f}%")
            st.metric("TVaR/EL Multiple", f"{risk_results.tvar_99 / risk_results.expected_loss:.2f}x" if risk_results.expected_loss > 0 else "N/A")
        with ratio_col2:
            st.metric("Premium/EL", f"{premium_breakdown.tvar_multiple:.2f}x")
            st.metric("Loss Occurrence", f"{risk_results.occurrence_rate:.1%}")
    
    with col_right:
        st.markdown("### Exceedance Probability Curve")
        
        losses = sim_result.year_loss_table["loss"].values
        losses_sorted = np.sort(losses)[::-1]
        n = len(losses_sorted)
        exceedance_prob = np.arange(1, n + 1) / n
        
        fig_ep = go.Figure()
        
        fig_ep.add_trace(go.Scatter(
            x=losses_sorted,
            y=exceedance_prob,
            mode="lines",
            line=dict(color="#c9a962", width=2),
            fill="tozeroy",
            fillcolor="rgba(201, 169, 98, 0.1)",
            name="EP Curve",
        ))
        
        # Add VaR markers with rotated annotations to avoid overlap
        fig_ep.add_vline(
            x=risk_results.var_99, 
            line_dash="dash", 
            line_color="#e4c87a", 
            annotation_text="VaR 99%", 
            annotation_position="top",
            annotation_textangle=-45,
            annotation_font=dict(color="#e4c87a", size=11),
        )
        fig_ep.add_vline(
            x=risk_results.tvar_99, 
            line_dash="dash", 
            line_color="#ef4444",
            annotation_text="TVaR 99%", 
            annotation_position="top",
            annotation_textangle=-45,
            annotation_font=dict(color="#ef4444", size=11),
        )
        
        fig_ep = apply_dark_theme(fig_ep)
        fig_ep.update_layout(
            height=450,
            xaxis_title="Loss ($M)",
            yaxis_title="Exceedance Probability",
            yaxis_type="log",
            yaxis_range=[-4, 0],
            margin=dict(t=20),
        )
        
        st.plotly_chart(fig_ep, use_container_width=True)
    
    # Portfolio composition and scenario analysis
    st.markdown("---")
    st.markdown("### Portfolio Composition & Scenario Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📈 Sector Exposure", "🎯 Scenario Breakdown", "🏢 Company Details"])
    
    with tab1:
        # Sector exposure pie chart
        sector_exposure = portfolio.exposure_by_sector()
        
        fig_sector = go.Figure(data=[go.Pie(
            labels=list(sector_exposure.keys()),
            values=list(sector_exposure.values()),
            hole=0.5,
            marker=dict(colors=["#c9a962", "#e4c87a", "#9a7d3e", "#22c55e", "#f59e0b"]),
            textinfo="label+percent",
            textfont=dict(color="#f5f5f5"),
        )])
        
        fig_sector = apply_dark_theme(fig_sector)
        fig_sector.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )
        
        col_pie, col_stats = st.columns([1.5, 1])
        
        with col_pie:
            st.plotly_chart(fig_sector, use_container_width=True)
        
        with col_stats:
            summary = portfolio.summary()
            st.markdown("#### Portfolio Statistics")
            st.metric("Total Revenue", f"${summary['total_revenue_M']:,.0f}M")
            st.metric("Avg. AI Dependency", f"{summary['average_ai_dependency']:.1%}")
            st.metric("Avg. Risk Score", f"{summary['average_risk_score']:.2f}")
            st.metric("Max Company Exposure", f"${summary['max_exposure_M']:,.0f}M")
    
    with tab2:
        # Scenario contribution analysis
        if sim_result.scenario_losses:
            scenario_el = {
                name: float(np.mean(losses))
                for name, losses in sim_result.scenario_losses.items()
            }
            
            fig_scenario = go.Figure(data=[go.Bar(
                x=list(scenario_el.keys()),
                y=list(scenario_el.values()),
                marker_color="#c9a962",
                text=[f"${v:,.2f}M" for v in scenario_el.values()],
                textposition="outside",
                textfont=dict(color="#f5f5f5"),
            )])
            
            fig_scenario = apply_dark_theme(fig_scenario)
            fig_scenario.update_layout(
                height=400,
                xaxis_title="Scenario",
                yaxis_title="Expected Loss ($M)",
                xaxis_tickangle=-45,
            )
            
            st.plotly_chart(fig_scenario, use_container_width=True)
        else:
            st.info("Scenario breakdown not available.")
    
    with tab3:
        # Company details table
        company_data = []
        for c in portfolio.companies:
            company_data.append({
                "Company": c.name,
                "Sector": c.sector.value.replace("_", " ").title(),
                "Revenue ($M)": f"{c.revenue:,.0f}",
                "Exposure ($M)": f"{c.exposure:,.0f}",
                "AI Dependency": f"{c.ai_dependency_score:.0%}",
                "Autonomy": f"{c.autonomy_level:.0%}",
                "Safety Score": f"{c.safety_score:.0%}",
                "Risk Score": f"{c.risk_score:.2f}",
            })
        
        df_companies = pd.DataFrame(company_data)
        st.dataframe(
            df_companies,
            use_container_width=True,
            hide_index=True,
        )
    
    # Risk metrics summary
    st.markdown("---")
    st.markdown("### Risk Metrics Summary")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>VaR 99%</h4>
                <div class="value">${risk_results.var_99:,.2f}M</div>
                <div style="color: #666666; font-size: 0.8rem;">1-in-100 year loss</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with metrics_col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>VaR 99.5%</h4>
                <div class="value">${risk_results.var_995:,.2f}M</div>
                <div style="color: #666666; font-size: 0.8rem;">1-in-200 year loss</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with metrics_col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>TVaR 99.5%</h4>
                <div class="value">${risk_results.tvar_995:,.2f}M</div>
                <div style="color: #666666; font-size: 0.8rem;">Tail expectation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with metrics_col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h4>Maximum Loss</h4>
                <div class="value">${risk_results.max_loss:,.2f}M</div>
                <div style="color: #666666; font-size: 0.8rem;">Simulated worst case</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: NEW COMPANY SIMULATION
# =============================================================================

elif page == "🏢 New Company Simulation":
    st.markdown("# New Company Risk Assessment")
    st.markdown(
        '<p style="color: #a0a0a0; font-size: 1.1rem;">Calculate individual premium for a new company seeking AI risk coverage</p>',
        unsafe_allow_html=True,
    )
    
    st.markdown("---")
    
    # Company input form
    col_form, col_viz = st.columns([1, 1.5])
    
    with col_form:
        st.markdown("### Company Profile")
        
        company_name = st.text_input(
            "Company Name",
            value="Acme AI Corp",
            help="Name of the company seeking coverage",
        )
        
        sector = st.selectbox(
            "Industry Sector",
            options=[s.value.replace("_", " ").title() for s in Sector],
            index=0,
        )
        sector_enum = Sector(sector.lower().replace(" ", "_"))
        
        revenue = st.number_input(
            "Annual Revenue ($M)",
            min_value=10.0,
            max_value=10000.0,
            value=500.0,
            step=50.0,
            help="Company's annual revenue in millions",
        )
        
        st.markdown("---")
        st.markdown("### Risk Characteristics")
        
        ai_dependency = st.slider(
            "AI Dependency Score",
            min_value=0.1,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="How reliant is the company on AI systems? (0.1=minimal, 0.95=critical)",
        )
        
        autonomy_level = st.slider(
            "AI Autonomy Level",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Degree of autonomous AI decision-making (0.1=human-supervised, 0.9=fully autonomous)",
        )
        
        st.markdown("---")
        st.markdown("### Safety Tools & Controls")
        st.markdown(
            '<p style="color: #888; font-size: 0.85rem;">Select the AI safety tools deployed by the company. '
            'These reduce risk exposure and lower premiums.</p>',
            unsafe_allow_html=True,
        )
        
        # Get provider registry for dropdown options
        registry = get_registry()
        
        # Define display names and icons for each surface
        surface_config = {
            RiskSurface.PROMPT_INJECTION: ("🛡️ Prompt Injection Defense", "Detects and blocks prompt injection attacks"),
            RiskSurface.GATEWAY: ("🚪 AI Gateway", "Traffic control, rate limiting, routing"),
            RiskSurface.MONITORING: ("📊 Runtime Monitoring", "Observability and anomaly detection"),
            RiskSurface.OUTPUT_FILTERING: ("🔍 Output Filtering", "Content moderation and safety filtering"),
            RiskSurface.EVALS: ("🧪 Model Evaluations", "Quality and safety testing frameworks"),
            RiskSurface.DATA_QUALITY: ("📁 Data Quality", "Data validation and quality monitoring"),
            RiskSurface.ACCESS_CONTROL: ("🔐 Access Control", "Authentication and authorization"),
            RiskSurface.MODEL_GOVERNANCE: ("📋 Model Governance", "Versioning, lineage, and compliance"),
        }
        
        # Store selections
        safety_selections: dict[str, str | None] = {}
        
        # Create expandable section for safety tools
        with st.expander("🔧 Configure Safety Tools", expanded=True):
            # Create two columns for the safety tools
            safety_col1, safety_col2 = st.columns(2)
            
            surfaces = list(surface_config.keys())
            for i, surface in enumerate(surfaces):
                display_name, help_text = surface_config[surface]
                providers = ["None"] + registry.list_providers(surface)
                
                # Alternate between columns
                col = safety_col1 if i % 2 == 0 else safety_col2
                
                with col:
                    selection = st.selectbox(
                        display_name,
                        options=providers,
                        index=0,
                        help=help_text,
                        key=f"safety_{surface.value}",
                    )
                    
                    if selection != "None":
                        safety_selections[surface.value] = selection
        
        # Build safety profile from selections
        if safety_selections:
            safety_profile = build_profile_from_selections(safety_selections)
            safety_score = safety_profile.overall_score
        else:
            safety_profile = SafetyProfile.empty()
            safety_score = 0.0
        
        # Show computed safety score
        n_tools = len(safety_selections)
        if n_tools > 0:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, rgba(201,169,98,0.1), rgba(201,169,98,0.05)); 
                     padding: 1rem; border-radius: 8px; border: 1px solid rgba(201,169,98,0.3); margin-top: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <span style="color: #a0a0a0; font-size: 0.85rem;">Safety Tools Deployed</span>
                            <div style="color: #c9a962; font-size: 1.5rem; font-weight: 600;">{n_tools} / {len(surface_config)}</div>
                        </div>
                        <div style="text-align: right;">
                            <span style="color: #a0a0a0; font-size: 0.85rem;">Computed Safety Score</span>
                            <div style="color: #c9a962; font-size: 1.5rem; font-weight: 600;">{safety_score:.1%}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("💡 Select safety tools above to reduce risk and premiums")
        
        st.markdown("---")
        
        simulation_mode = st.radio(
            "Calculation Mode",
            ["Quick Estimate", "Full Simulation"],
            horizontal=True,
            help="Quick estimate uses analytical approximations; Full simulation runs Monte Carlo",
        )
        
        calculate_btn = st.button("⚡ Calculate Premium", use_container_width=True)
    
    with col_viz:
        # Create company object with safety profile
        company = Company(
            name=company_name,
            revenue=revenue,
            ai_dependency_score=ai_dependency,
            autonomy_level=autonomy_level,
            _legacy_safety_score=safety_score,
            sector=sector_enum,
            safety_profile=safety_profile if safety_selections else None,
        )
        
        # Risk radar chart
        st.markdown("### Risk Profile Visualization")
        
        fig_radar = go.Figure()
        
        categories = ["AI Dependency", "Autonomy", "Risk Score", "1 - Safety", "Exposure Ratio"]
        values = [
            ai_dependency,
            autonomy_level,
            company.risk_score,
            1 - safety_score,
            min(1.0, company.exposure / revenue),  # Normalized exposure
        ]
        values.append(values[0])  # Close the polygon
        categories_closed = categories + [categories[0]]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories_closed,
            fill="toself",
            fillcolor="rgba(201, 169, 98, 0.2)",
            line=dict(color="#c9a962", width=2),
            name="Risk Profile",
        ))
        
        fig_radar = apply_dark_theme(fig_radar)
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    gridcolor="rgba(42,42,42,0.4)",
                    linecolor="#2a2a2a",
                    tickfont=dict(color="#a0a0a0"),
                ),
                angularaxis=dict(
                    gridcolor="rgba(42,42,42,0.4)",
                    linecolor="#2a2a2a",
                    tickfont=dict(color="#a0a0a0"),
                ),
                bgcolor="rgba(15,15,15,0.6)",
            ),
            height=350,
            showlegend=False,
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Quick stats
        st.markdown("### Exposure Analysis")
        
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        
        with stats_col1:
            st.metric("Calculated Exposure", f"${company.exposure:,.0f}M")
        
        with stats_col2:
            st.metric("Risk Score", f"{company.risk_score:.2f}")
        
        with stats_col3:
            exposure_ratio = company.exposure / revenue if revenue > 0 else 0
            st.metric("Exposure/Revenue", f"{exposure_ratio:.1%}")
    
    # Calculate premium when button clicked
    if calculate_btn:
        st.markdown("---")
        st.markdown("## Premium Calculation Results")
        
        with st.spinner("Calculating premium..."):
            calculator = IndividualPremiumCalculator(
                config=DEFAULT_CONFIG,
                n_simulation_years=5000 if simulation_mode == "Full Simulation" else 1000,
                seed=seed,
            )
            
            if simulation_mode == "Quick Estimate":
                result = calculator.calculate_quick_premium(company)
            else:
                result = calculator.calculate_premium(company)
        
        # Premium results
        st.markdown("---")
        
        result_col1, result_col2 = st.columns([1, 1])
        
        with result_col1:
            st.markdown(
                f"""
                <div class="premium-highlight" style="text-align: center;">
                    <h4 style="color: #a0a0a0; margin-bottom: 0.5rem;">TECHNICAL PREMIUM</h4>
                    <div style="font-size: 3rem; color: #c9a962; font-family: 'JetBrains Mono', monospace; font-weight: 600;">
                        ${result.standalone_premium:,.2f}M
                    </div>
                    <div style="color: #a0a0a0; margin-top: 0.5rem;">
                        Rate on Line: {result.rate_on_line:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.markdown("### Premium Components")
            
            # Premium breakdown chart
            components = {
                "Expected Loss": result.expected_loss_component,
                "Ambiguity Load": result.ambiguity_load_component,
                "Expense Load": result.expense_load_component,
            }
            
            fig_breakdown = go.Figure(data=[go.Pie(
                labels=list(components.keys()),
                values=list(components.values()),
                hole=0.6,
                marker=dict(colors=["#c9a962", "#e4c87a", "#f59e0b"]),
                textinfo="label+percent",
                textfont=dict(color="#f5f5f5"),
            )])
            
            fig_breakdown = apply_dark_theme(fig_breakdown)
            fig_breakdown.update_layout(
                height=300,
                showlegend=False,
                annotations=[dict(
                    text=f"${result.standalone_premium:,.1f}M",
                    x=0.5, y=0.5,
                    font_size=16,
                    font_color="#c9a962",
                    showarrow=False,
                )],
            )
            
            st.plotly_chart(fig_breakdown, use_container_width=True)
        
        with result_col2:
            st.markdown("### Risk Metrics")
            
            metrics_data = [
                ("Expected Loss (EL)", f"${result.standalone_expected_loss:,.2f}M", "Pure premium base"),
                ("VaR 99%", f"${result.standalone_var_99:,.2f}M", "1-in-100 year loss"),
                ("TVaR 99%", f"${result.standalone_tvar_99:,.2f}M", "Expected tail loss"),
                ("Loss Cost", f"{result.loss_cost:.2f}%", "EL as % of exposure"),
            ]
            
            for label, value, description in metrics_data:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h4>{label}</h4>
                        <div class="value" style="font-size: 1.5rem;">{value}</div>
                        <div style="color: #666666; font-size: 0.75rem;">{description}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        # Show mitigation impact if safety tools are selected
        if safety_selections and safety_profile.measures:
            st.markdown("---")
            st.markdown("### 🛡️ Safety Tools Impact Analysis")
            st.markdown(
                '<p style="color: #a0a0a0;">How your selected safety tools reduce risk exposure across different catastrophe scenarios</p>',
                unsafe_allow_html=True,
            )
            
            # Get scenarios and compute mitigation factors
            scenario_gen = ScenarioGenerator()
            scenarios = scenario_gen.get_predefined_scenarios()
            mitigation_engine = MitigationEngine()
            
            # Compute mitigation for each scenario
            mitigation_data = []
            for scenario in scenarios:
                analysis = mitigation_engine.analyze_mitigation(scenario, safety_profile)
                mitigation_data.append({
                    "Scenario": scenario.name,
                    "Mitigation Factor": analysis["mitigation_factor"],
                    "Risk Reduction": analysis["total_reduction"],
                    "Surfaces Covered": f"{analysis['n_covered_surfaces']}/{analysis['n_relevant_surfaces']}",
                    "Coverage Gaps": ", ".join(analysis["coverage_gaps"]) if analysis["coverage_gaps"] else "None",
                })
            
            mitigation_col1, mitigation_col2 = st.columns([1.2, 1])
            
            with mitigation_col1:
                # Mitigation bar chart
                scenario_names = [d["Scenario"].replace(" ", "\n") for d in mitigation_data]
                reductions = [d["Risk Reduction"] * 100 for d in mitigation_data]
                
                fig_mitigation = go.Figure()
                
                fig_mitigation.add_trace(go.Bar(
                    y=scenario_names,
                    x=reductions,
                    orientation="h",
                    marker=dict(
                        color=reductions,
                        colorscale=[[0, "#3d3d3d"], [0.5, "#c9a962"], [1, "#22c55e"]],
                        line=dict(color="#c9a962", width=1),
                    ),
                    text=[f"{r:.0f}%" for r in reductions],
                    textposition="outside",
                    textfont=dict(color="#c9a962"),
                ))
                
                fig_mitigation = apply_dark_theme(fig_mitigation)
                fig_mitigation.update_layout(
                    title=dict(
                        text="Risk Reduction by Scenario",
                        font=dict(color="#f5f5f5", size=14),
                    ),
                    xaxis=dict(
                        title="Risk Reduction (%)",
                        range=[0, 100],
                        gridcolor="rgba(42,42,42,0.4)",
                    ),
                    yaxis=dict(
                        tickfont=dict(size=10),
                    ),
                    height=350,
                    margin=dict(l=10, r=80, t=40, b=40),
                )
                
                st.plotly_chart(fig_mitigation, use_container_width=True)
            
            with mitigation_col2:
                st.markdown("#### Coverage Details")
                
                # Show per-surface effectiveness
                surface_scores = safety_profile.surface_scores()
                if surface_scores:
                    for surface, score in sorted(surface_scores.items(), key=lambda x: -x[1]):
                        # Get display name
                        display_name = surface.value.replace("_", " ").title()
                        # Color based on effectiveness
                        color = "#22c55e" if score >= 0.7 else "#c9a962" if score >= 0.5 else "#ef4444"
                        
                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; 
                                 padding: 0.5rem 0; border-bottom: 1px solid #2a2a2a;">
                                <span style="color: #a0a0a0; font-size: 0.9rem;">{display_name}</span>
                                <span style="color: {color}; font-weight: 600;">{score:.0%}</span>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                
                # Show uncovered surfaces as a warning
                uncovered = safety_profile.uncovered_surfaces
                # Remove "other" from uncovered as it's not typically needed
                uncovered = {s for s in uncovered if s != RiskSurface.OTHER}
                
                if uncovered:
                    st.markdown("#### ⚠️ Coverage Gaps")
                    uncovered_names = [s.value.replace("_", " ").title() for s in uncovered]
                    st.warning(f"Consider adding: {', '.join(uncovered_names[:4])}")
        
        # Safety investment analysis
        st.markdown("---")
        st.markdown("### Safety Investment ROI Analysis")
        st.markdown(
            '<p style="color: #a0a0a0;">See how improving safety practices can reduce your premium and unlock capital for AI safety investments</p>',
            unsafe_allow_html=True,
        )
        
        safety_col1, safety_col2 = st.columns([1, 1.5])
        
        with safety_col1:
            safety_improvement = st.slider(
                "Safety Score Improvement",
                min_value=0.05,
                max_value=0.30,
                value=0.10,
                step=0.05,
                help="How much could safety practices improve?",
            )
            
            # Calculate safety benefit
            benefit = estimate_safety_investment_benefit(
                company=company,
                safety_improvement=safety_improvement,
                calculator=calculator,
            )
            
            st.markdown(
                f"""
                <div class="metric-card" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(15, 15, 15, 0.9) 100%);">
                    <h4>Annual Premium Savings</h4>
                    <div class="value" style="color: #22c55e;">${benefit['premium_reduction']:,.2f}M</div>
                    <div style="color: #22c55e; font-size: 0.9rem;">↓ {benefit['reduction_percentage']:.1f}% reduction</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            st.metric(
                "New Safety Score",
                f"{benefit['improved_safety_score']:.0%}",
                delta=f"+{safety_improvement:.0%}",
            )
            
            st.metric(
                "New Premium",
                f"${benefit['improved_premium']:,.2f}M",
                delta=f"-${benefit['premium_reduction']:,.2f}M",
            )
        
        with safety_col2:
            # Safety investment curve
            improvements = np.linspace(0.05, 0.35, 7)
            premiums = []
            
            for imp in improvements:
                if company.safety_score + imp <= 1.0:
                    b = estimate_safety_investment_benefit(company, imp, calculator)
                    premiums.append(b["improved_premium"])
                else:
                    premiums.append(premiums[-1] if premiums else result.standalone_premium)
            
            fig_safety = go.Figure()
            
            fig_safety.add_trace(go.Scatter(
                x=[f"+{i:.0%}" for i in improvements],
                y=premiums,
                mode="lines+markers",
                line=dict(color="#22c55e", width=3),
                marker=dict(size=10, color="#22c55e"),
                name="Premium",
            ))
            
            # Add current premium line
            fig_safety.add_hline(
                y=result.standalone_premium,
                line_dash="dash",
                line_color="#ef4444",
                annotation_text="Current Premium",
                annotation_font=dict(color="#ef4444"),
            )
            
            fig_safety = apply_dark_theme(fig_safety)
            fig_safety.update_layout(
                height=350,
                xaxis_title="Safety Score Improvement",
                yaxis_title="Premium ($M)",
                showlegend=False,
            )
            
            st.plotly_chart(fig_safety, use_container_width=True)
        
        # Actuarial notes
        st.markdown("---")
        with st.expander("📝 Actuarial Notes & Methodology"):
            st.markdown(
                """
                #### Premium Calculation Methodology
                
                The technical premium is calculated using the following formula:
                
                **Premium = Expected Loss + Ambiguity Load + Expense Load**
                
                Where:
                - **Expected Loss (EL)**: Mean annual loss from Monte Carlo simulation
                - **Ambiguity Load**: α × TVaR₉₉% (compensates for parameter uncertainty)
                - **Expense Load**: ε × EL (operating costs)
                
                #### Key Assumptions
                
                1. **Heavy-Tailed Distributions**: AI catastrophe losses follow Pareto/Lognormal distributions
                2. **Correlation Structure**: Losses propagate through the AI supply chain dependency graph
                3. **No Historical Data**: Premium includes substantial ambiguity loading due to lack of loss experience
                4. **Safety Credits**: Higher safety scores reduce both exposure and tail severity
                
                #### Risk Factors
                
                - **AI Dependency**: Scales base exposure with operational reliance on AI
                - **Autonomy Level**: Amplifies exposure (autonomous systems have catastrophic failure modes)
                - **Safety Score**: Mitigates exposure (max 50% reduction from excellent practices)
                
                #### Rationale for Ambiguity Loading
                
                Unlike traditional lines with credible loss history, AI catastrophe risks have:
                - No historical loss data to calibrate frequencies
                - Unknown severity distributions
                - Potential for regime changes at capability thresholds
                
                The ambiguity load (typically 50% of TVaR) provides capital adequacy margin for this
                fundamental uncertainty about model parameters.
                """
            )


# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666666; padding: 2rem;">
        <p style="font-size: 0.9rem; color: #a0a0a0;">
            AI Catastrophe Risk Pricing Engine | Quantifying Systemic AI Risk
        </p>
        <p style="font-size: 0.75rem;">
            Built for actuaries and risk managers navigating the frontier of AI risk transfer
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
