# 🏗️ Singapore Water Potability Pipeline  (Snowflake Medallion Architecture & Data Product)

## 📖 About This Project

This project was developed for the **IT3383 - Data Processing on Big Data** assignment, focusing on building a complete data engineering solution using Snowflake and the Medallion Architecture framework.

The solution addresses water quality analysis, potability prediction, and data visualization for drinking water safety monitoring.

---

## 🎯 Problem Statement

Access to safe drinking water is essential to health and a basic human right. This project analyzes water quality data to:

- Analyse 3,276 water bodies across 9 different quality factors
- Predict water potability for Singapore water over 3 years
- Develop a data product to explain findings through visualizations

---

## 🏗️ Medallion Architecture

The Medallion Architecture was implemented in Snowflake with three core layers:

| Layer | Purpose |
|-------|---------|
| **Bronze** | Raw data ingestion with metadata tracking (LOAD_TIMESTAMP, LOAD_DATE) |
| **Silver** | Data cleaning, NULL handling, validation, and transformation |
| **Gold** | Aggregated tables and Materialized Views for analytics-ready data |

### Key Features
- **Time Travel:** 7-day data retention with UNDROP and BEFORE queries
- **Materialized Views:** Auto-refresh for zero processing lag
- **Aggregation Tables:** Pre-calculated metrics for dashboard queries
- **Metadata Tracking:** Data lineage with timestamps

---

## 📊 Datasets Used

| Dataset | Source | Description |
|---------|--------|-------------|
| `water_potability.csv` | Kaggle | Potability indicator (1 = Potable, 0 = Not potable) for corresponding water samples | 3,276 water bodies with 9 quality parameters (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity) 
| `SingaporeDrinkingwaterqualitydatasets.csv` | data.gov.sg (PUB) | Singapore drinking water quality data from 2019 to 2022, containing 80+ water quality parameters including pH, Conductivity, Total Dissolved Solids, Turbidity, and various chemical contaminants |

---

## 📈 Data Product (Streamlit Dashboard)

An interactive Streamlit dashboard connected to Snowflake's Gold Layer provides:

| Section | Features |
|---------|----------|
| **Global Controls** | Potability filter, Parameter range slider, Export options |
| **Key Metrics** | Safe Water %, Top Positive/Negative Factors, Best WHO Compliance |
| **EDA Tab** | 6 interactive charts (distributions, WHO compliance, trends, correlation, anomaly detection) |
| **Advanced Analytics** | Correlation matrix, Correlation explorer, Anomaly detection with IQR method |
| **Predictive Assessment** | Random Forest model with performance metrics and feature importance |

### Key Insights
- **39%** of water bodies are safe for consumption
- **Hardness** has highest WHO compliance at **100%**
- **Organic Carbon** has strongest negative correlation with potability
- Model achieves **67.84% accuracy** with **86% recall** for non-potable detection

---

## 💻 Tech Stack

| Component | Technology |
|-----------|------------|
| Data Platform | Snowflake |
| Data Architecture | Medallion Architecture (Bronze, Silver, Gold) |
| Dashboard Framework | Streamlit |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Random Forest) |
| Data Processing | Python, NumPy, Pandas |

---

## 💡 Key Recommendations

1. **Enhance treatment processes** to improve the 39% potability rate
2. **Prioritize WHO compliance gaps** (Conductivity, Chloramines, Sulfates)
3. **Maintain Singapore's excellent water quality** practices
4. **Automate anomaly detection** for proactive monitoring
5. **Address model bias** using SMOTE for class imbalance


