import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectFromModel
from numpy import sort
import seaborn as sns
def feature_selection_with_thresholds(xgb_model_fs, df_train, df_test, target_for_train, target_for_test):
    """
    Perform feature selection with different thresholds using an XGBoost model.

    The function trains an XGBoost model on the training data and iterates over different
    feature importances thresholds. For each threshold, it selects the features using the
    threshold and trains a new XGBoost model on the selected features. The performance of
    each model is evaluated using the F1 score. The function returns a dataframe with the
    evaluation results, a dataframe with the feature importances sorted by thresholds,
    and a line plot showing the F1 score evolution with the number of features.

    Args:
        xgb_model_fs: The XGBoost model used for feature selection.
        df_train (pd.DataFrame): The training data.
        df_test (pd.DataFrame): The test data.
        target_for_train (array-like): The target labels for the training data.
        target_for_test (array-like): The target labels for the test data.

    Returns:
        tuple: A tuple containing the following:
            - feature_list (pd.DataFrame): Dataframe with the evaluation results including
              the threshold, number of features, and F1 score.
            - thresholds_df (pd.DataFrame): Dataframe with the feature importances sorted by thresholds.
            - g (matplotlib.pyplot.Axes): Line plot showing the F1 score evolution with the number of features.

    Notes:
        - The xgb_model_fs should be an instance of XGBoost model already initialized with the desired parameters.
        - The df_train and df_test should be pandas DataFrames with the same column order.
        - The target_for_train and target_for_test should be array-like objects with the target labels.
        - The evaluation is performed using the F1 score weighted average.
        - The feature_list dataframe contains the evaluation results including the threshold,
          number of features selected, and F1 score.
        - The thresholds_df dataframe contains the feature importances sorted by thresholds.
        - The line plot shows the F1 score evolution with the number of features.
    """
    feature_list = []
    #Train model
    
    model_params = xgb_model_fs.get_params()
    xgb_model_fs.fit(df_train, target_for_train)
    thresholds = sort(xgb_model_fs.feature_importances_).tolist()
    # Find index of first non-zero element
    index = thresholds.index(next((x for x in thresholds if x != 0), None))
    thresholds = thresholds[index:]
    for thresh in thresholds:
        # Select features using threshold
        selection = SelectFromModel(xgb_model_fs, threshold=thresh, prefit=True)
        select_df_train = selection.transform(df_train)
        selection_model = XGBClassifier(**model_params)
        selection_model.fit(select_df_train, target_for_train)

        # Evaluate model
        select_df_test = selection.transform(df_test)
        predictions = selection_model.predict(select_df_test)
        f1 = f1_score(target_for_test, predictions, average='weighted')
        feature_list.append({'Threshold': thresh,
                        'Number of features': select_df_train.shape[1],
                        'f1 score': f1})
        print(f"Thresh={thresh}, n={select_df_train.shape[1]}, f1 score weighted: {f1}")

    feature_list = pd.DataFrame(feature_list)
    #feature_list.to_pickle(r'feature_selection_df_2')

    thresholds_sorted_id = xgb_model_fs.feature_importances_.argsort()
    thresholds_df = pd.DataFrame(xgb_model_fs.feature_importances_[thresholds_sorted_id],index=list(df_test.columns[thresholds_sorted_id]))
    #new_features_2 = thresholds_df_2[9:].index.values.tolist()
    g = sns.lineplot(data=feature_list, x="Number of features", y="f1 score",color='blue')
    g.set_xlabel('Number of features', fontsize = 18, labelpad=20)
    g.set_xlim(0,len(feature_list)+1)
    g.set_ylabel('Weighted f1 score', fontsize = 18, labelpad=20)
    g.set_title("f1 score evolution with number of features",fontsize=25);

    return feature_list, thresholds_df, g