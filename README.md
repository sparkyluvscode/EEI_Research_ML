Research Project under the mentorship of Abdiel Rivera, PhD in Electrical Engineering @Uconn

# Beyond Euler: An Explainable AI Framework for Predicting Pasta Buckling

This repository contains the dataset and Python analysis code for the research paper, "Beyond Euler: An Explainable Machine Learning Framework for Predicting and Interpreting Buckling Instabilities in Non-Ideal Materials."

We aim to answer the question: "Can we use machine learning to accurately predict the critical buckling load of pasta columns based on physical and environmental parameters?"

## Project Description

This project uses an XGBoost machine learning model to predict the critical buckling load of pasta columns from geometric features. It addresses the limitations of Euler's classical buckling formula for non-ideal materials. The analysis also employs SHAP (SHapley Additive exPlanations) to interpret the model's predictions, providing insights into the underlying physics.

## Files in this Repository

1.  **`final_eei_data.csv`**: The complete dataset containing 147 experimental samples of pasta buckling tests.
2.  **`final_analysis.py`**: The Python script used to perform the entire analysis.

## Requirements

To run the analysis, you will need Python 3 and the following libraries:

* pandas
* xgboost
* scikit-learn
* shap
* matplotlib

You can install these with pip or pip3 (mac):
`pip install pandas xgboost scikit-learn shap matplotlib`

## How to Run

1.  Clone this repository or download the files.
2.  Ensure `final_analysis.py` and `final_eei_data.csv` are in the same folder.
3.  Open a terminal or command prompt, navigate to that folder, and run the following command:

    `python final_analysis.py`

The script will print the final R² and RMSE values to the console and save the figures (`predicted_vs_actual.png` and `shap_summary_plot.png`) to the folder.
