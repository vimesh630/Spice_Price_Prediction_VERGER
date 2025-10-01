# 🌿 Cinnamon Price Forecasting System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-brightgreen.svg)](https://github.com/yourusername/cinnamon-forecasting/graphs/commit-activity)

**Advanced LSTM-based price forecasting system for Sri Lankan cinnamon market with hyperparameter tuning, confidence intervals, and smart alerts**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-demo)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Modes](#-usage-modes)
- [Output Examples](#-output-examples)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Configuration](#%EF%B8%8F-configuration)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

The **Cinnamon Price Forecasting System** is a comprehensive machine learning solution designed to predict cinnamon prices across different grades and regions in Sri Lanka. Built with state-of-the-art LSTM neural networks, the system provides accurate forecasts with statistical confidence intervals and actionable business insights.

### 🎬 Demo

![Forecast Dashboard](output.png)
*Interactive forecast dashboard with confidence intervals and smart alerts*

---

## ✨ Features

### 🤖 **Advanced Machine Learning**
- **LSTM Neural Networks** with 12-month lookback sequence
- **Automated Hyperparameter Tuning** using Keras Tuner (30-100 trials)
- **Ensemble Predictions** with multiple model architectures
- **Feature Engineering**: Lag features, rolling averages, seasonal patterns
- **Model Performance**: R² Score > 0.85, MAE < 150 LKR

### 📊 **Comprehensive Analytics**
- **95% & 68% Confidence Intervals** for uncertainty quantification
- **Prediction Confidence Scores** (0-100%) based on model reliability
- **Smart Alerts System**: Critical warnings, opportunities, recommendations
- **Regional Price Deviation Analysis** with arbitrage detection
- **Volatility Risk Assessment**: High/Medium/Low risk categorization

### 📈 **Three Forecasting Modes**

#### Mode 1: Regional Analysis
Forecast all grades for a specific region
**Input: Region+Forecast Period**
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
- **Excel Workbooks** with multiple sheets:
  - Summary Dashboard with charts
  - Smart Alerts & Insights
  - Monthly Forecasts with confidence intervals
  - Grade-wise & Region-wise summaries
- **PDF Executive Summaries** (2-3 pages)
- **Interactive Visualizations**: Line charts, bar charts, trend analysis
- **Color-coded Risk Indicators**: Visual risk assessment

### 🎨 **Enhanced User Experience**
- **Progress Bars** with ETA for long-running forecasts
- **Interactive CLI** with guided input validation
- **Fuzzy Matching** for region/grade name suggestions
- **Auto-complete** for common selections
- **Error Recovery** with graceful handling
- **Comprehensive Logging** for debugging

### 🔔 **Smart Insights Engine**
- **Critical Alerts**: High volatility (>20%), steep declines (>10%)
- **Warnings**: Elevated risk, negative trends
- **Opportunities**: Strong growth, buy signals, arbitrage gaps
- **Recommendations**: Portfolio advice, risk management strategies

---

## 🏗️ System Architecture
