# 🌿 Cinnamon Price Forecasting System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/yourusername/cinnamon-forecasting/graphs/commit-activity)

**Validated GRU-based price forecasting system for the Sri Lankan cinnamon market, delivering exceptional accuracy with automated hyperparameter tuning and smart alerts.**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Performance & Demo](#-performance--demo)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [System Architecture](#-system-architecture)
- [Key Data Insights](#-key-data-insights)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Modes](#-usage-modes)
- [Performance & Demo](#-performance--demo)
- [Project Structure](#-project-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The **Cinnamon Price Forecasting System** is a comprehensive machine learning solution designed to predict cinnamon prices across different grades and regions in Sri Lanka. Built with a state-of-the-art **Gated Recurrent Unit (GRU) neural network**, the system provides accurate forecasts with statistical confidence intervals and actionable business insights.

---

## ✨ Features

### 🤖 **Advanced Machine Learning**
- **Optimized GRU Neural Networks** with a 12-month lookback sequence
- **Automated Hyperparameter Tuning** using the **Optuna** framework (Best architecture is a Stacked GRU model)
- **Feature Engineering**: Lag features (1, 3, 6, 12 months) and rolling averages (3, 6, 12 months) for key variables

### 📊 **Performance Metrics (Test Set)**
The model significantly exceeded the initial performance targets (R² > 0.85, MAE < 150 LKR).

| Metric | Target | Achieved | Status |
| :--- | :--- | :--- | :--- |
| **R² Score** | > 0.85 | **0.9859** | ✅ EXCELLENT |
| **MAE (LKR)** | < 150 | **67.34** | ✅ EXCELLENT |
| **RMSE (LKR)** | N/A | **101.93** | N/A |

### 📈 **Three Forecasting Modes**

#### Mode 1: Regional Analysis
Forecast all grades for a specific region
**Input: Region + Forecast Period**
**Output: All grades price forecasts with comparisons**

#### Mode 2: Grade Regional Analysis
Analyze one grade across all regions with price deviation
**Input: Grade + Forecast Period**
**Output: Regional price differences and arbitrage opportunities**

#### Mode 3: Comprehensive Market Report
Complete market analysis for all grade-region combinations
**Input: Forecast Period Only**
**Output: Full market intelligence report (40+ combinations)**

### 📄 **Professional Reporting**
- **Excel Workbooks** with multiple sheets: Summary Dashboard, Smart Alerts, Monthly Forecasts, Grade-wise & Region-wise summaries
- **PDF Executive Summaries** (2-3 pages)

---

## 🔎 Key Data Insights

The initial data analysis reveals important characteristics of the Sri Lankan cinnamon market:

| Grade | Mean Regional Price (LKR) | Price Range (Min - Max) | Observation |
| :--- | :--- | :--- | :--- |
| **alba** | **4145.98** | 2566.50 - 6420.00 | Highest mean price and widest distribution, indicating premium status. |
| **h\_faq** | **2286.17** | 1407.50 - 3324.50 | Lowest mean price, representing the entry-level or lowest quality grade. |

**Highly Correlated Feature Pairs (|correlation| > 0.7):**
* `Exchange_Rate` ↔ `Fuel_Price`: 0.980
* `Regional_Price` ↔ `National_Price`: 0.978
* `Month_num` ↔ `Quarter`: 0.971

---

## 🚀 Performance & Demo

### 🎬 Demo


*Interactive forecast dashboard with confidence intervals and smart alerts*

---

## 💻 Quick Start

Follow these steps to set up and run the forecasting system:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/yourusername/cinnamon-forecasting.git](https://github.com/yourusername/cinnamon-forecasting.git)
    cd cinnamon-forecasting
    ```

2.  **Set up the Python Environment:**
    (Requires Python 3.8+ and the relevant libraries)
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Main Notebook:**
    Execute the `LSTM_Cinnamon.ipynb` notebook to preprocess data, perform hyperparameter tuning, train the final model, and generate initial forecasts.
    ```bash
    # Open the notebook in your preferred environment (e.g., Jupyter, VS Code)
    # Execute all cells sequentially.
    ```

---

## 🧠 System Architecture

The core of the system is a time-series deep learning architecture:

1.  **Data Preparation**: Clean data, encode categorical variables (`Grade`, `Region`), and engineer **lag** and **rolling mean** features.
2.  **Scaling**: Apply `StandardScaler` to features and the target variable (`Regional_Price`).
3.  **Sequence Creation**: Transform the time-series data into supervised sequences with a **12-month lookback** (e.g., 12 months of data to predict the 13th month's price).
4.  **Model (GRU/LSTM)**: A Stacked Recurrent Neural Network is used for training.
5.  **Forecasting & Reporting**: The trained model generates future price points, and a subsequent module processes these into professional reports and alerts (e.g., `Report_Generator.ipynb`).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
