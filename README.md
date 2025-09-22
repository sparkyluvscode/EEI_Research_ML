Research Project under the mentorship of Abdiel Rivera, PhD in Electrical Engineering @Uconn

## Beyond Euler: An Explainable Machine Learning Framework for Predicting and Interpreting Buckling Instabilities in Non-Ideal Materials

#### Pranil Raichurav \*<sup>1</sup>, Brandon Yee <sup>1</sup>, Dr. Abdiel Rivera <sup>1</sup>

<sup>1</sup> Department of Electrical and Computer Engineering, University of Connecticut, 06269, CT, USA

Correspondence: pranil.raichura@gmail.com

This repository contains the dataset and Python analysis code for the research paper, "Beyond Euler: An Explainable Machine Learning Framework for Predicting and Interpreting Buckling Instabilities in Non-Ideal Materials."

We aim to answer the question: "Can we use machine learning to accurately predict the critical buckling load of pasta columns based on physical and environmental parameters?"

## Project Description

This project uses an XGBoost machine learning model to predict the critical buckling load of pasta columns from geometric features. It addresses the limitations of Euler's classical buckling formula for non-ideal materials. The analysis also employs SHAP (SHapley Additive exPlanations) to interpret the model's predictions, providing insights into the underlying physics.

## Files in this Repository

1.  **`buckling_data.csv`**: The complete dataset containing 147 experimental samples of pasta buckling tests.
2.  **`main.py`**: The Python script used to perform the entire analysis.
3. **`/results/`**: The analysis results directory

## How to Run

```
git clone https://github.com/sparkyluvscode/EEI_Research_ML.git
cd EEI_Research_ML.git

pip install -r requirements.txt

python main.py
```
