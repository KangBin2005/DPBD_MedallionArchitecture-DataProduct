import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)
import numpy as np
from datetime import datetime

# 1. Setup with enhanced styling
st.set_page_config(
    layout="wide",
    page_title="Water Quality Dashboard",
    initial_sidebar_state="expanded"
)

# Custom CSS for better presentation
st.markdown("""
    <style>
        /* Main container styling */
        .main {
            padding: 0rem 1rem;
        }
        
        /* Title styling */
        h1 {
            color: #1f77b4;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            border-bottom: 3px solid #1f77b4;
        }
        
        /* Subheader styling */
        h2, h3 {
            color: #2c3e50;
            font-weight: 600 !important;
            margin-top: 1rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        /* Metric container styling */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 1rem !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #1f77b4;
            transition: transform 0.2s;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }
        
        /* Custom card styling for info boxes */
        .custom-info {
            background-color: #e7f3fe;
            border-left: 6px solid #2196F3;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        .custom-success {
            background-color: #d4edda;
            border-left: 6px solid #28a745;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        .custom-warning {
            background-color: #fff3cd;
            border-left: 6px solid #ffc107;
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 10px;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #e9ecef;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: white !important;
        }
        
        /* Divider styling */
        hr {
            margin: 1.5rem 0;
            border: 0;
            height: 2px;
            background: linear-gradient(to right, #1f77b4, transparent);
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #f8f9fa;
            border-radius: 8px;
            font-weight: 600;
        }
        
        /* Code block styling */
        .stCodeBlock {
            border-radius: 8px;
            border: 1px solid #dee2e6;
        }
        
        /* Button styling */
        .stButton button {
            background-color: #1f77b4;
            color: white;
            border-radius: 8px;
            font-weight: 500;
            transition: all 0.3s;
        }
        
        .stButton button:hover {
            background-color: #155a8a;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        /* Select box styling */
        .stSelectbox {
            background-color: white;
            border-radius: 8px;
        }
        
        /* Checkbox styling */
        .stCheckbox {
            background-color: #f8f9fa;
            padding: 0.5rem;
            border-radius: 8px;
        }
        
        /* Plot container styling */
        .stImage, .stPlotlyChart {
            background-color: white;
            border-radius: 10px;
            padding: 0.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 10px;
            overflow: hidden;
        }
        
        /* Success/Info/Warning/Error blocks */
        .stAlert {
            border-radius: 10px !important;
            border-left: 6px solid !important;
        }
        
        /* Divider between sections */
        .section-divider {
            margin: 2rem 0;
            border: 0;
            height: 1px;
            background: linear-gradient(to right, transparent, #1f77b4, transparent);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# CONNECT TO SNOWFLAKE
# ==========================================
conn = st.connection("snowflake")
session = conn.session()

# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================
@st.cache_data
def get_kaggle_data():
    df = session.sql("SELECT * FROM PARROT_db.GOLD.WATER_PORTABILITY_ANALYTICS").to_pandas()
    df.columns = [col.upper() for col in df.columns]
    return df.fillna(df.median(numeric_only=True))

@st.cache_data
def get_potability_distribution():
    df = session.sql("SELECT * FROM PARROT_db.GOLD.PORTABILITY_DISTRIBUTION").to_pandas()
    return df

@st.cache_data
def get_who_compliance():
    df = session.sql("SELECT * FROM PARROT_db.GOLD.WHO_COMPLIANCE ORDER BY COMPLIANCE_PCT").to_pandas()
    return df

@st.cache_data
def get_singapore_trends():
    df = session.sql("SELECT * FROM PARROT_db.GOLD.SG_WATER_TRENDS ORDER BY YEAR").to_pandas()
    df.columns = [col.upper() for col in df.columns]
    return df

@st.cache_data
def get_yearly_trends():
    df = session.sql("SELECT * FROM PARROT_db.GOLD.YEARLY_TRENDS").to_pandas()
    df.columns = [col.upper() for col in df.columns]
    return df

# Load all data
df_kag = get_kaggle_data()
df_potability = get_potability_distribution()
df_who = get_who_compliance()
df_sg = get_singapore_trends()
df_yearly = get_yearly_trends()

# ==========================================
# FEATURE LABELS AND CONSTANTS (DEFINED EARLY)
# ==========================================
FEATURE_LABELS = {
    'PH': 'pH Value',
    'HARDNESS': 'Hardness (mg/L)',
    'SOLIDS': 'Total Dissolved Solids (mg/L)',
    'CHLORAMINES': 'Chloramines (mg/L)',
    'SULFATE': 'Sulfate (mg/L)',
    'CONDUCTIVITY': 'Conductivity (μS/cm)',
    'ORGANIC_CARBON': 'Organic Carbon (mg/L)',
    'TRIHALOMETHANES': 'Trihalomethanes (μg/L)',
    'TURBIDITY': 'Turbidity (NTU)'
}

# WHO Standards from Data Dictionary
WHO_STANDARDS = {
    'pH': '6.5 - 8.5',
    'Hardness': '≤ 500 mg/L',
    'TDS': '≤ 1000 mg/L',
    'Chloramines': '≤ 4.0 mg/L',
    'Sulfate': '≤ 250 mg/L',
    'Conductivity': '≤ 400 μS/cm',
    'Organic Carbon': '≤ 2.0 mg/L',
    'Trihalomethanes': '≤ 80 μg/L',
    'Turbidity': '≤ 5.0 NTU'
}

# ==========================================
# CALCULATE INSIGHTS (Using full dataset first)
# ==========================================
total = len(df_kag)
potable = len(df_kag[df_kag['POTABILITY'] == 1])
non_potable = len(df_kag[df_kag['POTABILITY'] == 0])
potable_pct = potable / total * 100 if total > 0 else 0

# Calculate correlation with potability
corr_with_potability = df_kag.corr()['POTABILITY'].drop('POTABILITY').sort_values(ascending=False)
top_positive = corr_with_potability.head(1)
top_negative = corr_with_potability.tail(1)

# Get WHO compliance insights
worst_compliance = df_who.loc[df_who['COMPLIANCE_PCT'].idxmin()]
best_compliance = df_who.loc[df_who['COMPLIANCE_PCT'].idxmax()]

# ==========================================
# SIDEBAR WITH GLOBAL FILTERS
# ==========================================
with st.sidebar:
    st.header("🎛️ Global Controls")
    st.markdown("---")
    
    # Initialize session state for filters
    if 'filters_applied' not in st.session_state:
        st.session_state.filters_applied = False
    if 'filtered_df' not in st.session_state:
        st.session_state.filtered_df = None
    
    # Global potability filter
    potability_filter = st.radio(
        "💧 Filter by Potability",
        ['All', 'Potable Only', 'Non-Potable Only'],
        index=0,
        help="Select which water samples to display"
    )
    
    st.markdown("---")
    
    # Parameter range filter
    st.subheader("📊 Parameter Range")
    range_param = st.selectbox(
        "Select Parameter",
        options=list(FEATURE_LABELS.keys()),
        format_func=lambda x: FEATURE_LABELS.get(x, x),
        help="Choose a parameter to filter by its value range"
    )
    
    # Get min/max values
    min_val = float(df_kag[range_param].min())
    max_val = float(df_kag[range_param].max())
    range_slider = st.slider(
        f"Range for {FEATURE_LABELS.get(range_param)}",
        min_val, max_val, (min_val, max_val),
        help="Adjust the range to filter data"
    )
    
    st.markdown("---")
    
    # Apply filters button
    col1, col2 = st.columns(2)
    with col1:
        apply_filters = st.button("🔄 Apply Filters", use_container_width=True)
    with col2:
        if st.button("↺ Reset", use_container_width=True):
            st.cache_data.clear()
            st.session_state.filters_applied = False
            st.session_state.filtered_df = None
            st.rerun()
    
    st.markdown("---")
    
    # Export section
    st.subheader("📥 Export Options")
    
    export_format = st.radio(
        "Export as:",
        ['CSV', 'Report'],
        index=0
    )
    
    if st.button("📊 Export Data", use_container_width=True):
        if export_format == 'CSV':
            export_df = st.session_state.filtered_df if st.session_state.filters_applied else df_kag
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"water_quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            # Generate report
            report_df = st.session_state.filtered_df if st.session_state.filters_applied else df_kag
            total_report = len(report_df)
            potable_report = len(report_df[report_df['POTABILITY'] == 1])
            non_potable_report = len(report_df[report_df['POTABILITY'] == 0])
            potable_pct_report = potable_report / total_report * 100 if total_report > 0 else 0
            
            report = f"""
            WATER QUALITY ANALYSIS REPORT
            =============================
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            FILTERS APPLIED
            ---------------
            Potability: {potability_filter}
            Parameter: {FEATURE_LABELS.get(range_param)}
            Range: {range_slider[0]:.2f} - {range_slider[1]:.2f}
            
            KEY METRICS
            -----------
            Total Samples: {total_report:,}
            Potable: {potable_report:,} ({potable_pct_report:.1f}%)
            Non-Potable: {non_potable_report:,} ({100-potable_pct_report:.1f}%)
            
            TOP FEATURES
            ------------
            Positive: {top_positive.index[0]} (r={top_positive.values[0]:.3f})
            Negative: {top_negative.index[0]} (r={top_negative.values[0]:.3f})
            
            WHO COMPLIANCE
            --------------
            Best: {best_compliance['PARAMETER']} ({best_compliance['COMPLIANCE_PCT']:.1f}%)
            Worst: {worst_compliance['PARAMETER']} ({worst_compliance['COMPLIANCE_PCT']:.1f}%)
            
            GENERATED BY
            ------------
            Water Quality Dashboard
            """
            
            st.download_button(
                label="📝 Download Report",
                data=report,
                file_name=f"water_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    st.markdown("---")
    
    # Data info
    if st.session_state.filtered_df is not None and st.session_state.filters_applied:
        st.caption(f"📊 Showing: {len(st.session_state.filtered_df):,} records")
    else:
        st.caption(f"📊 Total Records: {len(df_kag):,}")
    
    st.caption("💡 Tip: Use filters to explore specific data subsets")

# ==========================================
# APPLY GLOBAL FILTERS
# ==========================================
if apply_filters or st.session_state.filters_applied:
    filtered_df = df_kag.copy()
    
    # Apply potability filter
    if potability_filter == 'Potable Only':
        filtered_df = filtered_df[filtered_df['POTABILITY'] == 1]
    elif potability_filter == 'Non-Potable Only':
        filtered_df = filtered_df[filtered_df['POTABILITY'] == 0]
    
    # Apply range filter
    filtered_df = filtered_df[
        (filtered_df[range_param] >= range_slider[0]) & 
        (filtered_df[range_param] <= range_slider[1])
    ]
    
    st.session_state.filtered_df = filtered_df
    st.session_state.filters_applied = True
    df_display = filtered_df
else:
    df_display = df_kag

# ==========================================
# MAIN TITLE
# ==========================================
st.title("🌊 Water Quality & Potability Analysis Dashboard")

# ==========================================
# KEY INSIGHTS SECTION (Using filtered data)
# ==========================================
total_display = len(df_display)
potable_display = len(df_display[df_display['POTABILITY'] == 1])
non_potable_display = len(df_display[df_display['POTABILITY'] == 0])
potable_pct_display = potable_display / total_display * 100 if total_display > 0 else 0

# Recalculate correlation for filtered data
if len(df_display) > 1:
    corr_with_potability_display = df_display.corr()['POTABILITY'].drop('POTABILITY').sort_values(ascending=False)
    if len(corr_with_potability_display) > 0:
        top_positive_display = corr_with_potability_display.head(1)
        top_negative_display = corr_with_potability_display.tail(1)
    else:
        top_positive_display = pd.Series([0], index=['N/A'])
        top_negative_display = pd.Series([0], index=['N/A'])
else:
    top_positive_display = pd.Series([0], index=['N/A'])
    top_negative_display = pd.Series([0], index=['N/A'])

# Display key insights
st.markdown("### 📈 Key Insights at a Glance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "💧 Safe Water",
        f"{potable_pct_display:.1f}%",
        delta=f"{potable_display:,} of {total_display:,}",
        delta_color="normal"
    )

with col2:
    st.metric(
        "📈 Top Positive Factor",
        f"{FEATURE_LABELS.get(top_positive_display.index[0], top_positive_display.index[0])}",
        delta=f"r = {top_positive_display.values[0]:.3f}",
        delta_color="normal"
    )

with col3:
    st.metric(
        "📉 Top Negative Factor",
        f"{FEATURE_LABELS.get(top_negative_display.index[0], top_negative_display.index[0])}",
        delta=f"r = {top_negative_display.values[0]:.3f}",
        delta_color="inverse"
    )

with col4:
    st.metric(
        "🏆 Best WHO Compliance",
        f"{best_compliance['PARAMETER']}",
        delta=f"{best_compliance['COMPLIANCE_PCT']:.1f}%",
        delta_color="normal"
    )

# Enhanced insights
st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 1.5rem; 
            border-radius: 10px; 
            color: white; 
            margin: 1rem 0;">
    <h4 style="color: white; margin-top: 0;">📊 Key Findings</h4>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
        <div>✅ <strong>Only {:.1f}%</strong> of water bodies are safe for consumption ({:,} out of {:,})</div>
        <div>📈 <strong>{}</strong> has the strongest positive correlation (r = {:.3f})</div>
        <div>📉 <strong>{}</strong> has the strongest negative correlation (r = {:.3f})</div>
        <div>🏆 <strong>{}</strong> has the highest WHO compliance at {:.1f}%</div>
    </div>
</div>
""".format(
    potable_pct_display, potable_display, total_display,
    FEATURE_LABELS.get(top_positive_display.index[0], top_positive_display.index[0]), 
    top_positive_display.values[0] if len(top_positive_display) > 0 else 0,
    FEATURE_LABELS.get(top_negative_display.index[0], top_negative_display.index[0]), 
    top_negative_display.values[0] if len(top_negative_display) > 0 else 0,
    best_compliance['PARAMETER'], best_compliance['COMPLIANCE_PCT']
), unsafe_allow_html=True)

st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

# ==========================================
# TABS
# ==========================================
tab1, tab2, tab3 = st.tabs(["📊 Exploratory Data Analysis", "📈 Advanced Analytics", "🔮 Predictive Assessment"])

# ==========================================
# TAB 1: EDA (Using filtered data)
# ==========================================
with tab1:
    st.markdown("### 📊 Exploratory Data Analysis")
    st.markdown("""
    <div style="background-color: #e7f3fe; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <strong>🎯 Objective:</strong> Analyze 9 water quality factors across {:,} water bodies and identify 
        which factors impact water potability.
    </div>
    """.format(len(df_display)), unsafe_allow_html=True)
    
    # ==========================================
    # CHART 1: POTABILITY DISTRIBUTION
    # ==========================================
    st.markdown("### 1. Water Potability Distribution")
    st.markdown("*How many water bodies are safe for consumption?*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Use filtered data for pie chart
        potability_counts = df_display['POTABILITY'].value_counts().reset_index()
        potability_counts.columns = ['POTABILITY', 'COUNT']
        potability_counts['STATUS'] = potability_counts['POTABILITY'].map({1: 'Potable', 0: 'Non-Potable'})
        
        labels = potability_counts['STATUS']
        counts = potability_counts['COUNT']
        colors = ['#ff6b6b', '#51cf66']
        
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        wedges, texts, autotexts = ax.pie(
            counts, 
            labels=labels, 
            autopct='%1.1f%%', 
            colors=colors, 
            startangle=90,
            textprops={'fontsize': 12, 'fontweight': 'bold'}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(14)
            autotext.set_fontweight('bold')
        
        ax.set_title('Water Potability Distribution', fontweight='bold', fontsize=14, pad=20)
        
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1.5rem; border-radius: 10px;">
            <h4 style="margin-top: 0;">📊 Summary Statistics</h4>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        - **Total water bodies analyzed:** {total_display:,}
        - ✅ **Potable (safe):** {potable_display:,} ({potable_display/total_display*100:.1f}%)
        - ❌ **Non-Potable:** {non_potable_display:,} ({non_potable_display/total_display*100:.1f}%)
        """)
        
        if potable_pct_display < 50:
            st.error("🚨 **Critical:** Less than 50% of water bodies are potable. Immediate action required!")
        elif potable_pct_display < 70:
            st.warning("⚠️ **Moderate:** Less than 70% of water bodies are potable. Improvement needed.")
        else:
            st.success("✅ **Good:** More than 70% of water bodies are potable.")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # ==========================================
    # CHART 2: PARAMETER DISTRIBUTION (Using filtered data)
    # ==========================================
    st.markdown("### 2. Potable Water Distribution by Parameter")
    st.markdown("*How does each parameter affect water potability?*")
    
    df_potable = df_display[df_display['POTABILITY'] == 1]
    
    col_filter1, col_filter2, col_filter3 = st.columns([1, 1, 1])
    
    with col_filter1:
        selected_param = st.selectbox(
            "📊 Select Parameter",
            options=list(FEATURE_LABELS.keys()),
            format_func=lambda x: FEATURE_LABELS.get(x, x)
        )
    
    with col_filter2:
        chart_type = st.selectbox(
            "📈 Chart Type",
            options=['Scatter Plot', 'Histogram']
        )
    
    with col_filter3:
        show_comparison = st.checkbox("Show Non-Potable Comparison", value=True)
    
    potable_data = df_potable[selected_param].dropna()
    non_potable_data = df_display[df_display['POTABILITY'] == 0][selected_param].dropna()
    
    potable_mean = potable_data.mean() if len(potable_data) > 0 else 0
    potable_std = potable_data.std() if len(potable_data) > 0 else 0
    potable_median = potable_data.median() if len(potable_data) > 0 else 0
    potable_min = potable_data.min() if len(potable_data) > 0 else 0
    potable_max = potable_data.max() if len(potable_data) > 0 else 0
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        if chart_type == 'Scatter Plot':
            x = potable_data
            y = [1] * len(x)
            
            np.random.seed(42)
            y_jitter = np.array(y) + np.random.normal(0, 0.05, len(y))
            
            ax.scatter(x, y_jitter, alpha=0.6, s=50, c='#51cf66', edgecolors='black', linewidth=0.5)
            
            if show_comparison and len(non_potable_data) > 0:
                x_non = non_potable_data
                y_non = [0] * len(x_non)
                y_non_jitter = np.array(y_non) + np.random.normal(0, 0.05, len(y_non))
                ax.scatter(x_non, y_non_jitter, alpha=0.4, s=40, c='#ff6b6b', edgecolors='black', linewidth=0.5, label='Non-Potable')
            
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['Non-Potable', 'Potable'])
            ax.set_ylabel('Potability', fontsize=11)
            ax.set_xlabel(FEATURE_LABELS.get(selected_param, selected_param), fontsize=11)
            
            ax.axvline(x=potable_mean, color='green', linestyle='--', alpha=0.8, linewidth=2, label=f'Potable Mean: {potable_mean:.2f}')
            
            if show_comparison and len(non_potable_data) > 0:
                non_potable_mean = non_potable_data.mean()
                ax.axvline(x=non_potable_mean, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Non-Potable Mean: {non_potable_mean:.2f}')
                ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
            else:
                ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none')
            
            ax.set_title(f'Potable Water Distribution: {FEATURE_LABELS.get(selected_param, selected_param)}', fontweight='bold', fontsize=14)
            ax.grid(True, alpha=0.2, linestyle='--')
            
        else:
            ax.hist(potable_data, bins=30, alpha=0.7, color='#51cf66', edgecolor='black', linewidth=0.5, label='Potable')
            
            if show_comparison and len(non_potable_data) > 0:
                ax.hist(non_potable_data, bins=30, alpha=0.5, color='#ff6b6b', edgecolor='black', linewidth=0.5, label='Non-Potable')
            
            ax.axvline(x=potable_mean, color='green', linestyle='--', linewidth=2, label=f'Potable Mean: {potable_mean:.2f}')
            
            if show_comparison and len(non_potable_data) > 0:
                non_potable_mean = non_potable_data.mean()
                ax.axvline(x=non_potable_mean, color='red', linestyle='--', linewidth=2, label=f'Non-Potable Mean: {non_potable_mean:.2f}')
            
            ax.set_xlabel(FEATURE_LABELS.get(selected_param, selected_param), fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'Distribution of {FEATURE_LABELS.get(selected_param, selected_param)}', fontweight='bold', fontsize=14)
            ax.legend(frameon=True, facecolor='white', edgecolor='none')
            ax.grid(True, alpha=0.2, linestyle='--')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
            <h4>📊 Statistics</h4>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        **Parameter:** {FEATURE_LABELS.get(selected_param, selected_param)}
        - **Count:** {len(potable_data):,}
        - **Mean:** {potable_mean:.2f}
        - **Median:** {potable_median:.2f}
        - **Std Dev:** {potable_std:.2f}
        - **Range:** {potable_min:.2f} - {potable_max:.2f}
        """)
        
        if show_comparison and len(non_potable_data) > 0:
            st.markdown("---")
            st.markdown("**📊 Comparison with Non-Potable:**")
            
            non_potable_mean = non_potable_data.mean()
            diff = potable_mean - non_potable_mean
            
            st.markdown(f"**Potable Mean:** {potable_mean:.2f}")
            st.markdown(f"**Non-Potable Mean:** {non_potable_mean:.2f}")
            st.markdown(f"**Difference:** {diff:.2f}")
            
            if diff > 0:
                st.success(f"✅ Higher by {diff:.2f}")
            else:
                st.error(f"❌ Lower by {abs(diff):.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # ==========================================
    # CHART 3: WHO COMPLIANCE
    # ==========================================
    st.markdown("### 3. WHO Standards Compliance Check")
    st.markdown("*How many water bodies meet WHO recommended limits?*")
    
    col_filter, col_empty = st.columns([1, 3])
    
    with col_filter:
        selected_who_param = st.selectbox(
            "🔍 Filter by Parameter",
            options=['All'] + df_who['PARAMETER'].tolist()
        )
    
    if selected_who_param != 'All':
        df_who_filtered = df_who[df_who['PARAMETER'] == selected_who_param]
    else:
        df_who_filtered = df_who
    
    # Define param_units HERE so it's available everywhere
    param_units = {
        'pH': '',
        'Hardness': 'mg/L',
        'TDS': 'mg/L',
        'Chloramines': 'mg/L',
        'Sulfate': 'mg/L',
        'Conductivity': 'μS/cm',
        'Organic Carbon': 'mg/L',
        'Trihalomethanes': 'μg/L',
        'Turbidity': 'NTU'
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        df_who_sorted = df_who.sort_values('COMPLIANCE_PCT', ascending=True)
        
        # Define WHO standard ranges for each parameter
        who_ranges = {
            'pH': (6.5, 8.5),
            'Hardness': (None, 500),
            'TDS': (None, 1000),
            'Chloramines': (None, 4.0),
            'Sulfate': (None, 250),
            'Conductivity': (None, 400),
            'Organic Carbon': (None, 2.0),
            'Trihalomethanes': (None, 80),
            'Turbidity': (None, 5.0)
        }
        
        colors = []
        for _, row in df_who_sorted.iterrows():
            param = row['PARAMETER']
            avg_val = row['AVG_VALUE']
            
            # Check if selected parameter - highlight in blue
            if selected_who_param != 'All' and param == selected_who_param:
                colors.append('#0066cc')
            else:
                # Check if average value meets WHO standard
                meets_standard = True
                if param in who_ranges:
                    lower, upper = who_ranges[param]
                    if lower is not None and avg_val < lower:
                        meets_standard = False
                    if upper is not None and avg_val > upper:
                        meets_standard = False
                else:
                    meets_standard = False
                
                if meets_standard:
                    colors.append('#51cf66')  # Green - average meets WHO standard
                else:
                    colors.append('#ff6b6b')  # Red - average does NOT meet WHO standard
        
        bars = ax.barh(
            df_who_sorted['PARAMETER'],
            df_who_sorted['COMPLIANCE_PCT'],
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        
        for bar, pct in zip(bars, df_who_sorted['COMPLIANCE_PCT']):
            width = bar.get_width()
            ax.text(
                width + 1,
                bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        worst = df_who.loc[df_who['COMPLIANCE_PCT'].idxmin()]
        best = df_who.loc[df_who['COMPLIANCE_PCT'].idxmax()]
        
        ax.text(0.02, 0.02, f"⚠️ Worst: {worst['PARAMETER']} ({worst['COMPLIANCE_PCT']:.1f}%)", 
                transform=ax.transAxes, fontsize=10, color='red', verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.98, f"✅ Best: {best['PARAMETER']} ({best['COMPLIANCE_PCT']:.1f}%)", 
                transform=ax.transAxes, fontsize=10, color='green', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Compliance Percentage (%)', fontsize=12)
        ax.set_title('WHO Standards Compliance by Parameter', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 105)
        ax.grid(True, alpha=0.2, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        if selected_who_param != 'All':
            st.markdown(f"**📊 Details for: {selected_who_param}**")
            
            if not df_who_filtered.empty:
                row = df_who_filtered.iloc[0]
                
                unit = param_units.get(row['PARAMETER'], '')
                
                # Check if average meets WHO standard for status display
                param = row['PARAMETER']
                avg_val = row['AVG_VALUE']
                lower, upper = who_ranges.get(param, (None, None))
                
                meets_standard = True
                if lower is not None and avg_val < lower:
                    meets_standard = False
                if upper is not None and avg_val > upper:
                    meets_standard = False
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
                    <p><strong>Compliance:</strong> {row['COMPLIANCE_PCT']:.1f}%</p>
                    <p><strong>Average Value:</strong> {row['AVG_VALUE']:.2f} {unit}</p>
                    <p><strong>WHO Standard:</strong> {WHO_STANDARDS.get(row['PARAMETER'], 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
                
                if meets_standard:
                    st.success("✅ **Meets WHO standard**")
                else:
                    st.error("❌ **Does NOT meet WHO standard**")
        else:
            st.markdown("""
            <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
                <h4>📊 Summary</h4>
            """, unsafe_allow_html=True)
            
            avg_compliance = df_who['COMPLIANCE_PCT'].mean()
            st.metric("Average Compliance", f"{avg_compliance:.1f}%")
            
            # Count parameters meeting WHO standard based on average value
            meeting_count = 0
            for _, row in df_who.iterrows():
                param = row['PARAMETER']
                avg_val = row['AVG_VALUE']
                lower, upper = who_ranges.get(param, (None, None))
                
                meets_standard = True
                if lower is not None and avg_val < lower:
                    meets_standard = False
                if upper is not None and avg_val > upper:
                    meets_standard = False
                
                if meets_standard:
                    meeting_count += 1
            
            total_params = len(df_who)
            st.metric("Parameters Meeting Standard", f"{meeting_count}/{total_params}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Parameters NOT meeting WHO standards - now based on average value logic
        st.markdown("**⚠️ Parameters NOT meeting WHO standards:**")
        
        # Find parameters that do NOT meet WHO standard based on average value
        failing_params = []
        for _, row in df_who.iterrows():
            param = row['PARAMETER']
            avg_val = row['AVG_VALUE']
            lower, upper = who_ranges.get(param, (None, None))
            
            meets_standard = True
            if lower is not None and avg_val < lower:
                meets_standard = False
            if upper is not None and avg_val > upper:
                meets_standard = False
            
            if not meets_standard:
                failing_params.append(row)
        
        if len(failing_params) > 0:
            for row in failing_params:
                unit = param_units.get(row['PARAMETER'], '')
                st.error(f"🔴 **{row['PARAMETER']}** - Average {row['AVG_VALUE']:.2f} {unit} (WHO Standard: {WHO_STANDARDS.get(row['PARAMETER'], 'N/A')})")
        else:
            st.success("✅ All parameters meet WHO standards!")
        
        with st.expander("📋 WHO Standards Used"):
            for param, standard in WHO_STANDARDS.items():
                st.write(f"- **{param}:** {standard}")
    
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # ==========================================
    # CHART 4: SINGAPORE TRENDS
    # ==========================================
    st.markdown("### 4. Singapore Water Quality Trends (2019-2022)")
    st.markdown("*How has each water quality parameter changed over time?*")
    
    df_yearly_pivot = df_yearly.pivot(
        index='YEAR', 
        columns='WATER_QUALITY_PARAMETER', 
        values='AVG_VALUE'
    ).reset_index()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), facecolor='white')
    
    if 'PH VALUE' in df_yearly_pivot.columns:
        years = df_yearly_pivot['YEAR']
        values = df_yearly_pivot['PH VALUE']
        
        ax1.plot(years, values, marker='o', color='#2ecc71', linewidth=2, markersize=10)
        ax1.axhline(y=6.5, color='red', linestyle='--', alpha=0.5, label='WHO Min (6.5)')
        ax1.axhline(y=8.5, color='red', linestyle='--', alpha=0.5, label='WHO Max (8.5)')
        
        for i, (year, value) in enumerate(zip(years, values)):
            ax1.annotate(f'{value:.2f}', 
                        xy=(year, value), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='#2ecc71',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
        
        ax1.set_title('pH Value', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Year', fontsize=10)
        ax1.set_ylabel('pH', fontsize=10)
        ax1.legend(frameon=True, facecolor='white', edgecolor='none')
        ax1.grid(True, alpha=0.2, linestyle='--')
        ax1.set_xticks(years)
    
    if 'TOTAL DISSOLVED SOLID' in df_yearly_pivot.columns:
        years = df_yearly_pivot['YEAR']
        values = df_yearly_pivot['TOTAL DISSOLVED SOLID']
        
        ax2.plot(years, values, marker='o', color='#3498db', linewidth=2, markersize=10)
        ax2.axhline(y=500, color='green', linestyle='--', alpha=0.5, label='Desirable (500)')
        ax2.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='Max (1000)')
        
        for i, (year, value) in enumerate(zip(years, values)):
            ax2.annotate(f'{value:.2f}', 
                        xy=(year, value), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='#3498db',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
        
        ax2.set_title('Total Dissolved Solids', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Year', fontsize=10)
        ax2.set_ylabel('mg/L', fontsize=10)
        ax2.legend(frameon=True, facecolor='white', edgecolor='none')
        ax2.grid(True, alpha=0.2, linestyle='--')
        ax2.set_xticks(years)
    
    if 'CONDUCTIVITY' in df_yearly_pivot.columns:
        years = df_yearly_pivot['YEAR']
        values = df_yearly_pivot['CONDUCTIVITY']
        
        ax3.plot(years, values, marker='o', color='#e74c3c', linewidth=2, markersize=10)
        ax3.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='WHO Max (400)')
        
        for i, (year, value) in enumerate(zip(years, values)):
            ax3.annotate(f'{value:.2f}', 
                        xy=(year, value), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='#e74c3c',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
        
        ax3.set_title('Conductivity', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Year', fontsize=10)
        ax3.set_ylabel('μS/cm', fontsize=10)
        ax3.legend(frameon=True, facecolor='white', edgecolor='none')
        ax3.grid(True, alpha=0.2, linestyle='--')
        ax3.set_xticks(years)
    
    if 'TURBIDITY' in df_yearly_pivot.columns:
        years = df_yearly_pivot['YEAR']
        values = df_yearly_pivot['TURBIDITY']
        
        ax4.plot(years, values, marker='o', color='#f39c12', linewidth=2, markersize=10)
        ax4.axhline(y=5.0, color='red', linestyle='--', alpha=0.5, label='WHO Max (5.0)')
        
        for i, (year, value) in enumerate(zip(years, values)):
            ax4.annotate(f'{value:.2f}', 
                        xy=(year, value), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        va='bottom',
                        fontsize=9,
                        fontweight='bold',
                        color='#f39c12',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7))
        
        ax4.set_title('Turbidity', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Year', fontsize=10)
        ax4.set_ylabel('NTU', fontsize=10)
        ax4.legend(frameon=True, facecolor='white', edgecolor='none')
        ax4.grid(True, alpha=0.2, linestyle='--')
        ax4.set_xticks(years)
    
    plt.suptitle('Singapore Water Quality Trends (2019-2022)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    with st.expander("📋 View Detailed Yearly Data"):
        st.dataframe(df_yearly_pivot, use_container_width=True)
        
        st.markdown("**💡 Trend Insights:**")
        if len(df_yearly_pivot) > 1:
            for col in df_yearly_pivot.columns:
                if col != 'YEAR':
                    values = df_yearly_pivot[col].dropna()
                    if len(values) > 1:
                        change = values.iloc[-1] - values.iloc[0]
                        direction = "📈 increased" if change > 0 else "📉 decreased"
                        st.caption(f"- **{col}** {direction} by {abs(change):.2f} from {values.iloc[0]:.2f} to {values.iloc[-1]:.2f}")



# ==========================================
# TAB 2: ADVANCED ANALYTICS
# ==========================================
with tab2:
    st.markdown("### 📈 Advanced Analytics")
    st.markdown("""
    <div style="background-color: #e7f3fe; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <strong>🔬 Objective:</strong> Explore deeper relationships and detect anomalies in water quality data
    </div>
    """, unsafe_allow_html=True)
    
    # ==========================================
    # CHART 5: CORRELATION MATRIX
    # ==========================================
    st.markdown("### 1. Feature Correlation Matrix")
    st.markdown("*How are different water quality parameters related to each other?*")
    
    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    # Calculate correlation matrix for displayed data
    corr_matrix = df_display.drop('POTABILITY', axis=1).corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation Coefficient"},
        annot_kws={'size': 10}
    )
    ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Interactive correlation explorer
    with st.expander("🔍 Explore Specific Correlations", expanded=False):
        st.markdown("*Select two features to see their correlation and scatter plot*")
        
        col1, col2 = st.columns(2)
        with col1:
            corr_feature1 = st.selectbox(
                "Select first feature",
                options=list(FEATURE_LABELS.keys()),
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key='corr1'
            )
        with col2:
            corr_feature2 = st.selectbox(
                "Select second feature",
                options=list(FEATURE_LABELS.keys()),
                format_func=lambda x: FEATURE_LABELS.get(x, x),
                key='corr2'
            )
        
        if corr_feature1 and corr_feature2 and corr_feature1 != corr_feature2:
            corr_value = df_display[corr_feature1].corr(df_display[corr_feature2])
            
            # Show correlation metric
            st.metric(
                f"Correlation between {FEATURE_LABELS.get(corr_feature1)} and {FEATURE_LABELS.get(corr_feature2)}",
                f"{corr_value:.3f}",
                delta=f"{'Positive' if corr_value > 0 else 'Negative'} correlation",
                delta_color="normal" if corr_value > 0 else "inverse"
            )
            
            # Scatter plot with regression line
            fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
            
            # Color points by potability
            scatter = ax.scatter(
                df_display[corr_feature1], 
                df_display[corr_feature2],
                c=df_display['POTABILITY'],
                cmap='coolwarm',
                alpha=0.6,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
            
            # Add regression line
            z = np.polyfit(df_display[corr_feature1], df_display[corr_feature2], 1)
            p = np.poly1d(z)
            ax.plot(df_display[corr_feature1].sort_values(), 
                   p(df_display[corr_feature1].sort_values()), 
                   "r--", linewidth=2, label=f'Correlation: {corr_value:.3f}')
            
            ax.set_xlabel(FEATURE_LABELS.get(corr_feature1), fontsize=12)
            ax.set_ylabel(FEATURE_LABELS.get(corr_feature2), fontsize=12)
            ax.set_title(f'Relationship: {FEATURE_LABELS.get(corr_feature1)} vs {FEATURE_LABELS.get(corr_feature2)}', 
                        fontweight='bold', fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.2)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Potability (0=Non-Potable, 1=Potable)', rotation=270, labelpad=20)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # Add interpretation
            st.info(f"""
            💡 **Interpretation:**
            - Correlation coefficient of **{corr_value:.3f}** indicates a 
            **{'strong' if abs(corr_value) > 0.7 else 'moderate' if abs(corr_value) > 0.3 else 'weak'}** 
            {'positive' if corr_value > 0 else 'negative'} relationship
            - This means that as {FEATURE_LABELS.get(corr_feature1)} {'increases' if corr_value > 0 else 'decreases'}, 
              {FEATURE_LABELS.get(corr_feature2)} tends to {'increase' if corr_value > 0 else 'decrease'} as well
            """)
        elif corr_feature1 == corr_feature2:
            st.warning("⚠️ Please select two different features for correlation analysis.")
    
    st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)
    
    # ==========================================
    # CHART 6: ANOMALY DETECTION
    # ==========================================
    st.markdown("### 2. Anomaly Detection")
    st.markdown("*Identify unusual water quality measurements that deviate from normal patterns*")
    
    # Anomaly detection using IQR method
    def detect_anomalies(df, feature, threshold=1.5):
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        anomalies = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        return anomalies, lower_bound, upper_bound
    
    col1, col2 = st.columns(2)
    with col1:
        anomaly_feature = st.selectbox(
            "Select feature for anomaly detection",
            options=list(FEATURE_LABELS.keys()),
            format_func=lambda x: FEATURE_LABELS.get(x, x),
            key='anomaly'
        )
    with col2:
        anomaly_threshold = st.slider(
            "IQR Threshold (higher = less anomalies)",
            min_value=1.0,
            max_value=3.0,
            value=1.5,
            step=0.1,
            help="Higher threshold detects fewer anomalies (more conservative)"
        )
    
    if anomaly_feature:
        anomalies, lower_bound, upper_bound = detect_anomalies(
            df_display, anomaly_feature, anomaly_threshold
        )
        
        # Plot distribution with anomaly highlighting
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        # Main histogram
        n, bins, patches = ax.hist(
            df_display[anomaly_feature], 
            bins=50, 
            alpha=0.7, 
            color='#1f77b4', 
            edgecolor='black',
            linewidth=0.5
        )
        
        # Highlight anomaly region
        ax.axvline(lower_bound, color='red', linestyle='--', linewidth=2, 
                  label=f'Lower Bound: {lower_bound:.2f}')
        ax.axvline(upper_bound, color='red', linestyle='--', linewidth=2, 
                  label=f'Upper Bound: {upper_bound:.2f}')
        
        # Shade anomaly regions
        ax.axvspan(
            df_display[anomaly_feature].min(), 
            lower_bound, 
            alpha=0.2, 
            color='red',
            label='Anomaly Region'
        )
        ax.axvspan(
            upper_bound, 
            df_display[anomaly_feature].max(), 
            alpha=0.2, 
            color='red'
        )
        
        # Highlight individual anomalies
        if len(anomalies) > 0 and len(anomalies) < 100:  # Only show if not too many
            for val in anomalies[anomaly_feature].values:
                ax.axvline(val, color='orange', alpha=0.4, linewidth=1)
        
        ax.set_xlabel(FEATURE_LABELS.get(anomaly_feature), fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Anomaly Detection: {FEATURE_LABELS.get(anomaly_feature)}', 
                    fontweight='bold', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.2)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Display anomaly statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df_display):,}")
        with col2:
            st.metric("Anomalies Detected", f"{len(anomalies):,}")
        with col3:
            anomaly_pct = len(anomalies) / len(df_display) * 100 if len(df_display) > 0 else 0
            st.metric("Anomaly Percentage", f"{anomaly_pct:.2f}%")
        with col4:
            st.metric("Normal Range", f"{lower_bound:.2f} - {upper_bound:.2f}")
        
        # Show anomaly records if any exist
        if len(anomalies) > 0:
            st.markdown("---")
            st.markdown(f"**📋 Anomaly Records ({len(anomalies)} detected)**")
            
            with st.expander(f"🔍 View {len(anomalies)} Anomaly Records"):
                st.dataframe(
                    anomalies.sort_values(anomaly_feature),
                    use_container_width=True
                )
                
                # Option to export anomalies
                anomaly_csv = anomalies.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Anomalies as CSV",
                    data=anomaly_csv,
                    file_name=f"anomalies_{anomaly_feature}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.success("✅ No anomalies detected with current threshold!")
            

# ==========================================
# TAB 3: PREDICTIVE ASSESSMENT
# ==========================================
with tab3:
    st.markdown("### 🔮 Predictive Model: Random Forest Classifier")
    st.markdown("""
    <div style="background-color: #e7f3fe; padding: 1rem; border-radius: 8px; margin-bottom: 1.5rem;">
        <strong>🎯 Objective:</strong> Build a Random Forest model to predict if water is potable (1) or non-potable (0)
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; margin-bottom: 1rem;">
        <div style="background-color: #d4edda; padding: 0.5rem; border-radius: 8px;">
            ✅ Handles non-linear relationships well
        </div>
        <div style="background-color: #d4edda; padding: 0.5rem; border-radius: 8px;">
            📊 Provides feature importance scores
        </div>
        <div style="background-color: #d4edda; padding: 0.5rem; border-radius: 8px;">
            🚀 Works well with tabular data
        </div>
        <div style="background-color: #d4edda; padding: 0.5rem; border-radius: 8px;">
            🛡️ Robust to outliers and missing values
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    @st.cache_resource
    def train_model(df):
        X = df.drop(columns=['POTABILITY'])
        y = df['POTABILITY']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        return {
            'model': model,
            'features': X.columns,
            'y_test': y_test,
            'y_pred': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cm': confusion_matrix(y_test, y_pred)
        }
    
    results = train_model(df_kag)
    
    st.markdown("""
    <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
        <strong>⚙️ Model Configuration:</strong><br>
        Algorithm: Random Forest Classifier | Trees: 100 | Train: 80% | Test: 20% | Features: 9
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy", f"{results['accuracy']:.2%}")
    with col2:
        st.metric("📊 Precision", f"{results['precision']:.2%}")
    with col3:
        st.metric("📈 Recall", f"{results['recall']:.2%}")
    with col4:
        st.metric("📉 F1 Score", f"{results['f1']:.2%}")
    
    st.markdown("---")
    
    st.markdown("### 1. Confusion Matrix")
    st.markdown("*Shows where the model got predictions right vs wrong*")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
        sns.heatmap(
            results['cm'],
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            square=True,
            xticklabels=['Non-Potable', 'Potable'],
            yticklabels=['Non-Potable', 'Potable'],
            annot_kws={'size': 16, 'fontweight': 'bold'}
        )
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        st.pyplot(fig)
        plt.close()
    
    with col2:
        tn, fp, fn, tp = results['cm'].ravel()
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
            <h4>📊 Breakdown</h4>
        """, unsafe_allow_html=True)
        
        st.success(f"✅ True Positives: **{tp}**")
        st.caption("Correctly predicted potable")
        st.success(f"✅ True Negatives: **{tn}**")
        st.caption("Correctly predicted non-potable")
        st.error(f"❌ False Positives: **{fp}**")
        st.caption("Predicted potable but actually non-potable")
        st.error(f"❌ False Negatives: **{fn}**")
        st.caption("Predicted non-potable but actually potable")
        
        total_pred = len(results['y_test'])
        correct = tp + tn
        st.info(f"💡 **{correct}/{total_pred}** ({correct/total_pred*100:.1f}%) correctly predicted")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 2. Classification Report")
    st.markdown("*Detailed performance metrics for the Random Forest model*")
    
    report_text = classification_report(
        results['y_test'], 
        results['y_pred'], 
        target_names=['Non-Potable', 'Potable']
    )
    
    st.code(report_text, language='text')
    st.caption(f"📊 Based on {len(results['y_test'])} test samples")
    
    st.markdown("---")
    
    st.markdown("### 3. Feature Importance")
    st.markdown("*Which factors does the model consider most important for predicting potability?*")
    
    importance = pd.DataFrame({
        'Feature': [FEATURE_LABELS.get(f, f) for f in results['features']],
        'Importance': results['model'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        
        importance_plot = importance.sort_values('Importance', ascending=True)
        
        bars = ax.barh(
            importance_plot['Feature'],
            importance_plot['Importance'],
            color='#1f77b4',
            edgecolor='black',
            linewidth=0.5
        )
        
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=10,
                fontweight='bold'
            )
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='x', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px;">
            <h4>🏆 Top Features</h4>
        """, unsafe_allow_html=True)
        
        for idx, row in importance.head(5).iterrows():
            if row['Importance'] > 0.01:
                st.success(f"✅ **{row['Feature']}**: {row['Importance']:.3f}")
            else:
                st.info(f"➖ **{row['Feature']}**: {row['Importance']:.3f}")
        
        st.markdown("---")
        st.info("💡 **Insight:** Features with higher importance scores have stronger influence on potability predictions.")
        
        st.markdown("</div>", unsafe_allow_html=True)