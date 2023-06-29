import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_validate
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, classification_report, confusion_matrix
from scipy.stats import loguniform


def adjacent_accuracy_score(y_true, y_pred):
    """
    Compute the adjacent accuracy score between the true and predicted labels.

    The adjacent accuracy score is calculated as the ratio of the number of
    predictions that are adjacent (differ by 0 or 1) to the total number of predictions.

    Args:
        y_true (array-like of shape (n_samples,)): The true labels.
        y_pred (array-like of shape (n_samples,)): The predicted labels.

    Returns:
        float: The adjacent accuracy score.

    Notes:
        - The true and predicted labels should have the same length.
        - The labels are expected to be numeric or convertible to numeric.
        - The adjacent accuracy score ranges from 0 to 1, where 1 represents perfect agreement and 0 represents pure chance.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(np.abs(y_pred - y_true) <= 1) / len(y_pred)


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""
    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)

def evaluate_model_with_cv(X, y, model, n_splits, plot = False):
    """
    Evaluate a model using cross-validation and calculate various evaluation metrics.

    The function performs cross-validation by splitting the data into training and test sets
    for a specified number of splits. The model is trained on the training data and evaluated
    on the test data for each split. The evaluation metrics calculated include accuracy,
    Cohen's Kappa score, F1 score, adjacent accuracy, classification report, confusion matrix,
    and feature importances.

    Args:
        X (array-like of shape (n_samples, n_features)): The feature matrix.
        y (array-like of shape (n_samples,)): The target labels.
        model: The machine learning model to be evaluated.
        n_splits (int): The number of splits for cross-validation.

    Returns:
        tuple: A tuple containing the following:
            - evaluation_df (pd.DataFrame): Dataframe with the mean and standard deviation of the evaluation metrics.
            - confusion_matrices (list): List of confusion matrices for each split.
            - reports (list): List of classification reports for each split.
            - feature_importance_df (pd.DataFrame): Dataframe with the mean feature importances.

    Notes:
        - The feature matrix `X` and target labels `y` should have the same number of samples.
        - The model should have a `fit()` and `predict()` method compatible with scikit-learn.
        - The evaluation metrics are calculated using scikit-learn functions.
        - The evaluation_df dataframe contains the mean and standard deviation of the evaluation metrics across splits.
        - The feature_importance_df dataframe contains the mean feature importances across splits.
        - The feature importances are also plotted
    """
    # initialize StratifiedKFold object
    cv_1 = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1)
    
    # initialize lists to store evaluation metrics
    accuracies = []
    cohen_kappas = []
    f1_scores = []
    adjacent_accuracies = []
    reports = []
    confusion_matrices = []
    feature_importance_dfs = []
    
    # loop over the folds
    for train_idx, test_idx in cv_1.split(X, y):
        # split the data into training and test sets
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # fit the model on the training data
        model.fit(X_train, y_train)

        # make predictions on the test data
        y_pred = model.predict(X_test)
        predictions = [round(value) for value in y_pred]

        # evaluate the predictions
        accuracy = accuracy_score(y_test, predictions)
        cohen_kappa = cohen_kappa_score(y_test, predictions, weights='quadratic')
        f1 = f1_score(y_test, predictions, average='weighted')
        adjacent_accuracy = adjacent_accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        confusion_matrix_ = confusion_matrix(y_test, predictions)
        feature_importance = model.feature_importances_
        
        # append the evaluation metrics to the corresponding lists
        accuracies.append(accuracy)
        cohen_kappas.append(cohen_kappa)
        f1_scores.append(f1)
        adjacent_accuracies.append(adjacent_accuracy)
        reports.append(report)
        confusion_matrices.append(confusion_matrix_)
        feature_importance_dfs.append(pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance}).sort_values(by='Importance', ascending=False))
        
    # concatenate the feature importance dataframes into a single dataframe
    feature_importance_df = pd.concat(feature_importance_dfs).groupby('Feature').mean().sort_values(by='Importance', ascending=False)
    
    # Group by 'Feature' and calculate the mean importance
    feature_importance_df_mean = feature_importance_df.groupby('Feature').mean().reset_index()
    
    # calculate the mean and standard deviation of the evaluation metrics
    mean_accuracy = sum(accuracies) / n_splits
    std_accuracy = (sum((x - mean_accuracy) ** 2 for x in accuracies) / n_splits) ** 0.5
    
    mean_cohen_kappa = sum(cohen_kappas) / n_splits
    std_cohen_kappa = (sum((x - mean_cohen_kappa) ** 2 for x in cohen_kappas) / n_splits) ** 0.5
    
    mean_f1_score = sum(f1_scores) / n_splits
    std_f1_score = (sum((x - mean_f1_score) ** 2 for x in f1_scores) / n_splits) ** 0.5
    
    mean_adjacent_accuracy = sum(adjacent_accuracies) / n_splits
    std_adjacent_accuracy = (sum((x - mean_adjacent_accuracy) ** 2 for x in adjacent_accuracies) / n_splits) ** 0.5
    
    # create a dataframe to store the evaluation metrics
    evaluation_df = pd.DataFrame({'Metric': ['Accuracy', 'Cohen_s Kappa', 'F1 Score', 'Adjacent Accuracy'],
                                   'Mean': [mean_accuracy, mean_cohen_kappa, mean_f1_score, mean_adjacent_accuracy],
                                   'Std': [std_accuracy, std_cohen_kappa, std_f1_score, std_adjacent_accuracy]})
    if plot == True:
        # Plot the mean feature importances
        plt.figure(figsize=(15, 20))
        sns.barplot(x='Importance', y='Feature', data=feature_importance_df_mean.sort_values(by='Importance', ascending=False))
        plt.title('Mean Feature Importances')
        plt.xlabel('Mean Importance')
        plt.ylabel('Feature')
        plt.show()
        # return the evaluation metrics
    return evaluation_df, confusion_matrices, reports, feature_importance_df

def randomized_search_and_cv(model, param_distributions, data, target, scoring):
    """
    Perform randomized search cross-validation using a given model and parameter distributions.

    Parameters:
        model (estimator): The estimator object implementing the scikit-learn estimator interface.
        param_distributions (dict): Dictionary of parameter distributions for the randomized search.
        data (array-like): The input data for training and evaluation.
        target (array-like): The target variable for training and evaluation.
        scoring (dict or list): Scoring metric(s) to be used for evaluation during cross-validation.

    Returns:
        cv_results_df (DataFrame): Pandas DataFrame containing the cross-validation results.

    Performs randomized search cross-validation by shuffling the data, fitting the model on the training folds,
    and evaluating the model on the test fold. Prints the mean and standard deviation of the custom F1 scores
    and the best hyperparameters for each fold. Returns a DataFrame with the complete cross-validation results.

    Example usage:
        cv_results = perform_randomized_search_cv(model_LR, param_distributions, preprocess_dataframe(data), target_cat, scoring)
    """
    model_random_search = RandomizedSearchCV(model, 
                                             param_distributions=param_distributions, 
                                             n_iter=20,
                                             error_score=np.nan, 
                                             n_jobs=3, 
                                             verbose=1, 
                                             random_state=1, 
                                             refit='custom_f1',
                                             scoring=scoring,
                                             return_train_score=True)

    cv_results = cross_validate(model_random_search, 
                                data, target, 
                                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=1), 
                                n_jobs=3, 
                                scoring=scoring,
                                return_train_score=True,
                                return_estimator=True)

    cv_results_df = pd.DataFrame(cv_results)

    mean_custom_f1, std_custom_f1 = cv_results_df['test_custom_f1'].mean(), cv_results_df['test_custom_f1'].std()
    mean_custom_cohen_kappa, std_custom_cohen_kappa = cv_results_df['test_cohen_kappa_score'].mean(), cv_results_df['test_cohen_kappa_score'].std()
    mean_custom_adjacent_accuracy, std_adjacent_accuracy = cv_results_df['test_adjacent_accuracy_score'].mean(), cv_results_df['test_adjacent_accuracy_score'].std()


    print(f"'{model}' generalization scores:\n"
          f"Weighted f1: {mean_custom_f1:.3f} +/- {std_custom_f1:.3f}\n"
          f"Cohen Kappa: {mean_custom_cohen_kappa:.3f} +/- {std_custom_cohen_kappa:.3f}\n"
          f"Adjacent Accuracy: {mean_custom_adjacent_accuracy:.3f} +/- {std_adjacent_accuracy:.3f}\n")

    for cv_fold, estimator_in_fold in enumerate(cv_results_df["estimator"]):
        print(f"Best hyperparameters for fold {cv_fold + 1} with a best result of {estimator_in_fold.best_score_}")
        print(estimator_in_fold.best_params_)

    return cv_results_df