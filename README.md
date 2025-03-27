# Laundry-Trend-Analyzer

Laundry Transaction Forecasting and Clustering

Overview

This project utilizes time series forecasting and machine learning techniques to analyze laundry transaction data. It implements:

Prophet for time series forecasting

Random Forest Regressor for predictive modeling

K-Means Clustering for pattern identification in transactions

Features

✅ Time Series Forecasting: Predict future laundry transactions using Prophet✅ Machine Learning Regression: Estimate future transaction counts using Random Forest✅ Clustering Analysis: Identify transaction patterns using K-Means clustering✅ Data Visualization: Generate trend and cluster visualizations

Dependencies

Ensure you have the following Python libraries installed:

pip install pandas numpy matplotlib prophet scikit-learn openpyxl

Dataset

The input data is expected to be an Excel file (laundrybook.xlsx) containing laundry transaction records.

Expected Columns:

Year: Transaction year

Month: Transaction month (string format, e.g., "January")

Day: Transaction day

Date: Automatically generated from Year, Month, and Day

Installation

Clone this repository:

git clone https://github.com/yourusername/your-repo.git
cd your-repo

Install dependencies:

pip install -r requirements.txt

Run the script:

python forecasting.py

Usage

1️⃣ Prophet Forecasting

Splits data into training (past transactions) and testing (recent 30 days)

Predicts future laundry transactions

Computes Mean Absolute Error (MAE) for accuracy

Generates a forecast plot

2️⃣ Random Forest Regression

Trains a Random Forest model to predict transaction counts

Evaluates model performance using MAE

Generates predictions for the next 30 days

3️⃣ K-Means Clustering

Groups transaction data into 4 clusters based on transaction volume

Visualizes clusters in a scatter plot

Outputs

Forecast Plot: Prophet-predicted transaction trends

Cluster Visualization: K-Means clustering of transaction patterns

Prediction Table: Future transaction estimates using Random Forest

Example Output

Prophet Forecast Plot:



K-Means Clustering Visualization:



License

This project is licensed under the MIT License. See the LICENSE file for details.

Author

Created by Your Name. Feel free to contribute or reach out for questions!
