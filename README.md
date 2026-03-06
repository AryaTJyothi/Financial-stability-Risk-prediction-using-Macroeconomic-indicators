# Financial Stability Risk Prediction using Macroeconomic indicators

## Project Overview

This project predicts the probability of an economic recession using macroeconomic indicators such as GDP growth rate, inflation and unemployment rate. The machine learning model analyse historical economic data and estimates whether a country is likely to experience a recession.

## Project Objective

The objective of this project is to build a machine learning model that can identify potential recession signals from economic indicators. This helps in understanding economic trends and provides insights for economic forecasting.

## Dataset

The dataset used in this project is sourced from the **IMF World Economic Outlook (WEO)** database.

The dataset includes macroeconomic indicators such as:

* GDP Growth
* Lagged GDP values (previous years)
* Inflation Rate
* Country-wise economic indicators
* Historical macroeconomic data

Official Source: https://www.imf.org/en/Publications/WEO/weo-database

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Flask
* HTML 

## Machine Learning Model

The prediction model is built using the **XGBoost Classifier**.

The model analyzes economic indicators and predicts whether a recession is likely.

Model Output:

* **0 → Recession Likely**
* **1 → No Recession**
