# Depression Prediction

This is a machine learning project focused on analyzing and predicting the severity of depression using self-reported 1-year health data from a mental health observational study. The objective of this project is to replicate and extend the analysis performed in the research paper by Makhmutova et al. [1] and present a complete machine learning pipeline for depression prediction. There is also some MLOps being currently performed.

## Dataset

The dataset used in this project is obtained from the paper by Makhmutova et al. [1], which can be found [here](https://github.com/jloayza10/depression_prediction_project/tree/main/data/raw/makhmutova2021.pdf). It consists of self-reported health data collected over a 1-year period.

## Machine Learning 

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

The necessary notebooks used are oredered and numbered accordingly. If no numbering is present, the notebook presents a parallel study.

## Results for xbgoost model

| Metric     | Adjacent Accuracy | Balanced Accuracy| Cohen-Kappa score| Weighted F1-score |
| :----:     |    :----:         |   :----:         |     :----:       |     :----:        |
| My results |     0.885         |     0.525        |      0.617       |      0.522        |
| Paper      |     0.889         |     0.472        |      0.661       |      0.543        |

A 1-page pdf report with the main results and visualizations is available [here](https://github.com/jloayza10/depression_prediction_project/blob/main/1-page_Report.pdf).
### Valuable Takeaways

- Directory machine learning project structure: use notebooks for exploring, testing and visualizing data, while scripts contain reusable functions
- Manually kept track of datasets (processed or not, split or full dataset) and lists of columns (selected features, to be deleted...). Next step is to automate this function
- [Dataframe assign method](https://github.com/jloayza10/depression_prediction_project/tree/main/scripts/data_cleaning.py)
- Feature preprocessing (different imputation or scaling methods depending on the original feature distribution), feature engineering and feature creation
- [Cross-validation function](https://github.com/jloayza10/depression_prediction_project/tree/main/scripts/scoring_and_evaluation.py) which calculates various evaluation metrics (scores, confusion matrix, classification report and feature importance)
- [Nested cross-validation](https://github.com/jloayza10/depression_prediction_project/tree/main/scripts/scoring_and_evaluation.py) for hyperparameter tuning and generalization scores  with 'RandomizedSearchCV', 'cross_validate' and custom scoring with the 'make_scorer' function
- [Feature selection function](https://github.com/jloayza10/depression_prediction_project/tree/main/scripts/XGBoost_feature_selection.py) with thresholds based on feature importance of xbgboost classifier model
- [Hyperopt syntax for xgboost hyperparameter tuning](https://github.com/jloayza10/depression_prediction_project/tree/main/notebooks/4-XGBoost_tuning.ipynb)
- Use ChatGPT for function docstrings
- Github file size limitations (for trained models for example)

#### To be done

- Deal with class imbalance (with SMOTE or another method)
- Perform error analysis on final model
- Train a simpler, smaller or faster model
- Data allows to work on this problem as a regression problem as well
- Check the Git LFS strategy for large file uploading/tracking

## MLOps 

The following steps are performed as part of the MLOps project:
- Clone repository and work in Unix virtual machine
- Virtual environment with pipenv
- [Flask app and script .py file](https://github.com/jloayza10/depression_prediction_project/tree/main/web-service/predict.py) for new input data reading and feature processing
- Docker and gunicorn server deployment

### Valuable Takeaways

- Discover the pipenv virtual environment (with Pipfile and Pipfile.lock)
- Flask app syntax
- [Dockerfile](https://github.com/jloayza10/depression_prediction_project/tree/main/web-service/Dockerfile) creation and gunicorn terminal commands

#### To be done

- Continue learning about deployment, Flask, Docker theory
- Deploy model to an online app
- Feature store, feature pipeline
- Experiment tracking, model registry and versioning
- Workflow orchestration
- CI/CD and testing


## References

[1] Makhmutova, A. et al.(2021). "Prediction of self-reported depression scores using Person-Generated Health Data from a Virtual 1-Year Mental Health Observational Study"  Proceedings of the 2021 Workshop on Future of Digital Biomarkers, Pages 4–11.
