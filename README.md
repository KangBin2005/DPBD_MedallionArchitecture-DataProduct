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
| `water_potability.csv` | Kaggle | Potability indicator (1 = Potable, 0 = Not potable) for corresponding water samples with 9 water quality parameters (pH, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic Carbon, Trihalomethanes, Turbidity) for 3,276 water bodies
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

## 📁 Repository Structure

```text
DPBD_MedallionArchitecture-DataProduct/
│
├── 244423Q_IT3383_Assignment.sql
│ └── Complete SQL script containing Bronze, Silver, and Gold layer implementations
│
├── SingaporeDrinkingwaterqualitydatasets.csv
│ └── Singapore PUB drinking water quality data (2019 and 2022, 80+ parameters)
│
├── water_potability.csv
│ └── Kaggle dataset with 9 water quality parameters and potability indicator
│
├── IT3383_244423Q_DPBD_Assignment_Report.docx
│ └── Full assignment report with all SQL code, screenshots, and analysis
│
└── README.md
└── This file
```


---

### SQL File Contents (`244423Q_IT3383_Assignment.sql`)

| Layer | Contents |
|-------|----------|
| **Bronze** | Stage creation, file format, temporary tables, COPY INTO commands, timestamp tracking |
| **Silver** | NULL handling with COALESCE, symbolic value conversion (`<` values), text rejection, validation queries |
| **Gold** | Materialized views, aggregation tables (potability distribution, WHO compliance, yearly trends) |
| **Time Travel** | UNDROP and BEFORE queries with 7-day retention demonstration |

---

### Data Files

| File | Description |
|------|-------------|
| `SingaporeDrinkingwaterqualitydatasets.csv` | PUB water quality data (2019-2022, 80+ parameters including pH, Conductivity, TDS, Turbidity, Monochloramine) |
| `water_potability.csv` | Kaggle dataset (3,276 water bodies, 9 quality parameters, potability indicator) |

---

### Report File

| File | Description |
|------|-------------|
| `IT3383_244423Q_DPBD_Assignment_Report.docx` | Complete assignment documentation with DDL/DML statements, screenshot analysis, and key insights |
