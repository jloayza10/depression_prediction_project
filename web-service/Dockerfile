FROM python:3.8.13-slim

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "df_study", "cols_to_del.pkl", "selected_features_2.pkl", "feature_preprocessing.py", "predict.py", "xgboost_mod.bin", "final_preprocessor.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]