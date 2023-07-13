# Depression Prediction

This is a machine learning project focused on analyzing and predicting the severity of depression using self-reported 1-year health data from a mental health observational study. The objective of this project is to replicate and extend the analysis performed in the research paper by Makhmutova et al. [1] and present a complete machine learning pipeline for depression prediction. There is also some MLOps steps being currently performed.

## Dataset

The dataset used in this project is obtained from the paper by Makhmutova et al. [1], which can be found [here](https://github.com/jloayza10/depression_prediction_project/tree/main/data/raw/makhmutova2021.pdf). It consists of self-reported health data collected over a one-year period.

## Project Steps

The following steps are performed as part of this machine learning project:

- Data Reading
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Visualizations
- Feature Engineering
- Feature Selection
- Model Selection
- Hyperparameter Tuning for xgboost model
- Model Evaluation
- Cross-Validation and nested cross-validation

#### To be done:
- Deal with class imbalance
- Perform error analysis on final model
- Train a simpler, smaller or faster model
- Data allows to work on this problem as a regression problem as well 

The following steps are performed as part of the MLOps project:
- Clone repo and work in unix virtual machine
- virtual environment with pipenv
- Flask app
- Script .py file for input data reading and feature processing
- gunicorn server deployment

#### To be done:
- Deploy model to an online app
- Feature store
- Experiment tracking, model registry and versioning
- Workflow orchestration
- CI/CD and testing

## Results

| Metric     | Adjacent Accuracy | Balanced Accuracy| Cohen-Kappa score| Weighted F1-score |
| :----:     |    :----:         |   :----:         |     :----:       |     :----:        |
| My results |     0.885         |     0.525        |      0.617       |      0.522        |
| Paper      |     0.889         |     0.472        |      0.661       |      0.543        |

## References

[1] Makhmutova, A. et al.(2021). "Prediction of self-reported depression scores using Person-Generated Health Data from a Virtual 1-Year Mental Health Observational Study"  Proceedings of the 2021 Workshop on Future of Digital Biomarkers, Pages 4–11.
