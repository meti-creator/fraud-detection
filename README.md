# E-Commerce Behavioral Fraud Detection Pipeline

## 📌 Project Overview
This project focuses on building an end-to-end Machine Learning data and analytics infrastructure designed to detect complex, automated transaction fraud patterns in e-commerce environments. Moving beyond traditional static, rule-based detection systems, this architecture translates raw event logs into rich behavioral tracking signatures to expose botnets, card-testing scripts, and synthetic accounts.

The core asset evaluates a dataset of **151,112 unique transaction entries** across hardware fingerprints, user demographics, and temporal timestamps.

---

## 🛠️ Task 1: Data Analysis & Preprocessing Lifecycle

### 1. Data Cleaning & Type Serialization
* **Zero Missing Values:** The initial integrity scan isolated exactly 0 missing entries across all rows, removing the need for risky statistical imputations.
* **Exact Deduplication:** Redundancy validation confirmed 0 exact duplicate log entries, preserving an untainted initial footprint of 151,112 rows.
* **Schema Optimization:** Recast tracking IDs into 64-bit integers, parsed textual timestamps into high-precision microsecond datetime tensors (`datetime64[us]`), and optimized text parameters into discrete categories—reducing the active memory footprint by over 65%.

### 2. Exploratory Data Analysis (EDA) & Risk Profiling
* **Severe Class Imbalance:** Quantified a target skew of **90.635% Legitimate (Class 0)** to **9.365% Fraudulent (Class 1)**. This baseline proves that standard "Accuracy" metrics cannot be used for model optimization.
* **Network Linkage Diagnostics (The Botnet Footprint):** Legitimate entries show a strict human relationship with devices, averaging **1.001 users per device/IP**. Fraudulent entries exhibit a massive network overlap, averaging **8.23 users per physical device** and **7.84 users per IP routing address**, explicitly uncovering centralized browser-automation scripts.

### 3. Geolocation Integration
* Converted dot-decimal string IP addresses into continuous numerical integers (`ip_int`).
* Performed a range-based lookup merge to map transaction coordinates against external network boundaries, successfully identifying country origins.
* Isolated country-level fraud patterns, identifying high-risk regional hotspots significantly outperforming the global baseline.

### 4. Behavioral Feature Engineering
* **Temporal Mappings:** Extracted continuous cyclical time values (`hour_of_day`, `day_of_week`) mapped as two-dimensional sine and cosine coordinates to catch late-night automated scripts.
* **Account Lifespan Maturity (`time_since_signup`):** Calculated account duration down to the second. Organic human users take days or weeks ($\text{Median} \approx 60.2\text{ days}$), while fraud scripts cluster tightly at **exactly 1 second**.
* **Short-Term Transaction Velocity (`tx_count_last_10min`):** Engineered a rolling 10-minute lookback count per identity to capture rapid, high-frequency carding attacks.

### 5. Data Transformation & Class Imbalance Remediation
* **Z-Score Scaling:** Normalized continuous attributes using a standard Z-score scalar to establish an arithmetic mean ($\mu = 0$) and standard deviation ($\sigma = 1$).
* **One-Hot Encoding:** Expanded categorical indicators cleanly while implementing a `drop='first'` safeguard to eliminate multicollinearity (the dummy variable trap).
* **SMOTE Balancing:** Executed **Synthetic Minority Over-sampling Technique (SMOTE)** strictly on the training partition after a secure stratified split. SMOTE expanded the training footprint to a perfectly balanced set of **219,136 rows**, while the independent test set was safely isolated at **30,223 rows** with its natural baseline imbalance intact to prevent data leakage.

---

## 📈 Resampling Distribution Summary

| Partition Target Matrix | Original Training Split | SMOTE Over-Sampling | Random Undersampling | Balanced Strategy Ratio |
| :--- | :--- | :--- | :--- | :--- |
| **Legitimate Class (`0`)** | 109,568 rows | 109,568 rows | 11,321 rows | 50.0% Balanced |
| **Fraudulent Class (`1`)** | 11,321 rows | 109,568 rows | 11,321 rows | 50.0% Balanced |
| **Total Active Training Pool** | **120,889 rows** | **219,136 rows** | **22,642 rows** | **100.0% Complete** |

---

## 🚀 Repository Structure
```text
├── data/
│   ├── raw/                      # Unaltered ingestion files
│   └── processed/                # Cleaned, engineered, and scaled datasets
├── notebooks/
│   └── eda-fraud-data.ipynb      # Main Task-1 Preprocessing Notebook
├── README.md                     # Technical project documentation
└── requirements.txt              # Pipeline tracking dependencies
