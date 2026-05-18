# ⚽ TransferIQ

<div align="center">

### AI-Powered Football Transfer Market Intelligence Platform

Predicting football player market values using Machine Learning, advanced analytics, and performance-driven insights.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge\&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-orange?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?style=for-the-badge\&logo=streamlit)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</div>

---

# 📌 Overview

TransferIQ is an AI-driven football analytics platform designed to estimate and predict professional football player market values using Machine Learning models and statistical analysis.

The system analyzes player performance metrics such as:

* Goals
* Assists
* Minutes Played
* Age
* Position
* Match Performance
* Offensive Contribution
* Efficiency Metrics

Using advanced feature engineering and predictive modeling, TransferIQ generates realistic transfer market valuations and performance insights useful for:

* Football scouting
* Recruitment analysis
* Talent identification
* Transfer strategy evaluation
* Sports analytics research

---

# 🚀 Features

## 🌍 Transfermarkt Data Parsing

* Automated parsing of player names and market values from Transfermarkt.
* Data extraction and integration pipeline for real-world football transfer valuations.
* Supports large-scale football player market intelligence collection.

## 🗂️ Large Scale Football Dataset Integration

* Built using the 2022-2023 Football Player Dataset from Kaggle.
* Dataset contains 124+ columns of player statistics and metadata.
* Utilizes 50+ engineered and selected features for model training and valuat

## 📊 Machine Learning Based Valuation

* Predicts player market values using trained ML models.
* Uses football performance statistics and engineered features.
* Supports regression-based valuation systems.

## 📈 Advanced Analytics Dashboard

* Visual representation of player metrics.
* Correlation heatmaps.
* Feature importance analysis.
* Actual vs Predicted comparison.

## ⚡ Data Processing Pipeline

* Data cleaning and preprocessing.
* Feature selection and transformation.
* Dataset merging and statistical normalization.

## 🧠 Model Intelligence

* XGBoost-powered predictive modeling.
* Performance metric evaluation.
* Error analysis using RMSE, MAE, and R².

## 🌐 Interactive Frontend

* Streamlit-powered user interface.
* Dynamic player valuation predictions.
* Easy-to-use football analytics dashboard.

---

# 🏗️ Project Architecture

```text
Football Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
Machine Learning Model
       │
       ▼
Prediction Engine
       │
       ▼
Visualization Dashboard
```

---

# 🛠️ Tech Stack

| Category         | Technologies          |
| ---------------- | --------------------- |
| Programming      | Python                |
| Machine Learning | XGBoost, Scikit-learn |
| Data Processing  | Pandas, NumPy         |
| Visualization    | Matplotlib, Seaborn   |
| Frontend         | Streamlit             |
| Dataset Handling | CSV, Excel            |
| Model Storage    | Pickle (.pkl)         |

---

# 📂 Project Structure

```bash
TransferIQ/
│
├── app.py
├── main.ipynb
├── merged_dataset.csv
├── preprocessed_dataset.csv
├── player_market_values.csv
├── xgb_final_model.pkl
├── feature_importance.xlsx
├── player_data_with_predictions.xlsx
├── .gitignore
├── README.md
│
├── visualizations/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── actual_vs_predicted.png
│   └── residuals.png
│
└── datasets/
```

---

# 📊 Machine Learning Workflow

## 1️⃣ Data Collection

Football player statistics and market value datasets are collected and merged.

## 2️⃣ Data Preprocessing

* Handling missing values
* Feature encoding
* Data normalization
* Dataset cleaning

## 3️⃣ Feature Engineering

Creation of meaningful football analytics features such as:

* Goals per 90
* Assists per 90
* Contribution metrics
* Efficiency ratios

## 4️⃣ Model Training

Models trained using:

* XGBoost Regressor
* Ensemble techniques
* Performance optimization

## 5️⃣ Evaluation

Evaluation metrics include:

* RMSE
* MAE
* R² Score
* Residual Analysis

---

# 📸 Visualizations Included

✅ Feature Correlation Heatmap
✅ Feature Importance Graphs
✅ Actual vs Predicted Market Values
✅ Residual Error Analysis
✅ Market Value Distribution

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Fy1zN/TransferIQ.git
cd TransferIQ
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

```bash
streamlit run app.py
```

---

# 📈 Example Use Cases

* Football transfer market prediction
* AI-powered scouting systems
* Player valuation research
* Sports analytics dashboards
* Recruitment strategy optimization
* Performance intelligence systems

---

# 🔮 Future Enhancements

* Real-time football API integration
* Deep Learning valuation systems
* Transfer recommendation engine
* Player similarity analysis
* Injury impact prediction
* Team chemistry prediction
* Market trend forecasting
* Cloud deployment support

---

# 📚 Research & Learning Applications

TransferIQ can be used for:

* Sports AI research
* Football analytics learning
* Machine Learning portfolio projects
* Final year academic projects
* Data science demonstrations
* Predictive analytics experimentation

---

# 🤝 Contributing

Contributions, ideas, and improvements are welcome.

```bash
git fork
Create feature branch
Commit changes
Submit pull request
```

---

# 📄 License

This project is licensed under the MIT License.

---

# 👨‍💻 Author

### Krish Malhotra

AI/ML Developer • Football Analytics Enthusiast • Full Stack Developer

---

<div align="center">

### ⭐ If you found this project useful, consider starring the repository.

</div>
