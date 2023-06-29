import pandas as pd
import numpy as np
def mathematical_transforms(df):
    """
    Performs mathematical transformations on the input DataFrame to create new features.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the original features.
    
    Returns:
        pandas.DataFrame: The DataFrame with new transformed features.
    """
    X = pd.DataFrame()  # dataframe to hold new features
    # Thresholds
    if "sleep_asleep_week_mean" in df.columns:
        X["sleep_threshold"] = (~((df["sleep_asleep_week_mean"]>300) & (df["sleep_asleep_week_mean"] < 500))).astype('float32') # original 200 and 500 #396
        
        if all(col in df.columns for col in ["bmi", "money", "sex", "age", "insurance"]):
    
            X["sleep_threshold_money"] = (X["sleep_threshold"]*df["money"]).astype('float32')
            X["sleep_threshold_bmi"] = (X["sleep_threshold"]*df["bmi"]).astype('float32')
            X["sleep_threshold_sex"] = (X["sleep_threshold"]*df["sex"]).astype('float32')
            X["sleep_threshold_age"] = (X["sleep_threshold"]*df["age"]).astype('float32')
            X["sleep_threshold_insurance"] = (X["sleep_threshold"]*df["insurance"]).astype('float32')
    if "sleep_in_bed_week_mean" in df.columns:
            
        X["in_bed_threshold"] = (~((df['sleep_in_bed_week_mean']>300) & (df['sleep_in_bed_week_mean'] < 600))).astype('float32')
        
    if "steps_awake_mean" in df.columns:
        X["steps_threshold"] = (df["steps_awake_mean"] < 8000).astype('float32') #[4330.470703125, 6535.171630859375, 9663.369140625] 6000
        if all(col in df.columns for col in ["bmi", "money", "sex", "age", "insurance"]):
    
            X["steps_threshold_money"] = (X["steps_threshold"]*df["money"]).astype('float32')
            X["steps_threshold_bmi"] = (X["steps_threshold"]*df["bmi"]).astype('float32')
            X["steps_threshold_sex"] = (X["steps_threshold"]*df["sex"]).astype('float32')
            X["steps_threshold_age"] = (X["steps_threshold"]*df["age"]).astype('float32')
            X["steps_threshold_insurance"] = (X["steps_threshold"]*df["insurance"]).astype('float32')
            #Indicator interaction
            X['steps_money_indicator'] = np.where((df['money'] < 1),
                                                  0,
                                                  X['steps_threshold'])
            X["steps_insurance_indicator"] = (X["steps_threshold"] * df["insurance"]).astype('float32')
    #Ratios
    if "sleep_asleep_week_mean" in df.columns:
        X["steps_sleep_ratio"] = (df["steps_awake_mean"]/ df["sleep_asleep_week_mean"]).astype('float32')
        
    

    #Interactions, polynomial features selected from a previous study on mutual information 
    if "bmi" in df.columns:
        X["bmi^2"] = (df["bmi"]**2).astype('float32')
        X["bmi^3"] = df["bmi"]**3
    
    if all(col in df.columns for col in ["bmi", "money"]):
    
        X["bmi_money"] = (df["bmi"]*df["money"]).astype('float32')
        X["bmi^2_money"] = ((df["bmi"]**2)*df["money"]).astype('float32')
        X["bmi_money^2"] = (df["bmi"]*(df["money"]**2)).astype('float32')
    
    if all(col in df.columns for col in ["bmi", "insurance"]):
        X["bmi_insurance"] = (df["bmi"]*df["money"]).astype('float32')
        X["bmi^2_insurance"] = ((df["bmi"]**2)*df["money"]).astype('float32')
        X["bmi_insurance^2"] = (df["bmi"]*(df["insurance"]**2)).astype('float32')
        
    if all(col in df.columns for col in ["bmi", "sleep_asleep_week_mean"]):
        X["bmi_sleep_asleep_week_mean"] = (df["bmi"]*df["sleep_asleep_week_mean"]).astype('float32')
    if all(col in df.columns for col in ["bmi", "sleep_in_bed_week_mean"]):
        X["bmi_sleep_in_bed_week_mean"] = (df["bmi"]*df["sleep_in_bed_week_mean"]).astype('float32')
    if all(col in df.columns for col in ["bmi", "steps_awake_mean"]):
        X["bmi_steps_awake_mean"] = (df["bmi"]*df["steps_awake_mean"]).astype('float32')
    if all(col in df.columns for col in ["bmi", "sleep_in_bed_mean_recent"]):
        X["bmi_sleep_in_bed_mean_recent"] = (df["bmi"]*df["sleep_in_bed_mean_recent"]).astype('float32')
    if all(col in df.columns for col in ["bmi", "sleep_ratio_asleep_in_bed_mean_recent"]):
        X["bmi_sleep_ratio_asleep_in_bed_mean_recent"] = (df["bmi"]*df["sleep_ratio_asleep_in_bed_mean_recent"]).astype('float32')
    
    
    
    if all(col in df.columns for col in ["sleep_asleep_week_mean", "money"]):
        X["sleep_asleep_week_mean_money"] = (df["sleep_asleep_week_mean"]*df["money"]).astype('float32')
    if all(col in df.columns for col in ["sleep_in_bed_week_mean", "money"]):
        X["sleep_in_bed_week_mean_money"] = (df["sleep_in_bed_week_mean"]*df["money"]).astype('float32')
    if all(col in df.columns for col in ["steps_awake_mean", "money"]):
        X["steps_awake_mean_money"] = (df["steps_awake_mean"]*df["money"]).astype('float32')


    return X

def counts(df):
    """
    Performs count-based feature engineering on the input DataFrame to create new features based on these counts.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the original features.
    
    Returns:
        pandas.DataFrame: The DataFrame with new count-based features.
    """
    X = pd.DataFrame()
    df_study = pd.read_pickle(r'../data/processed/df_study')
    #Intermediate column creation
    df['money_scaled'] = ((4-df_study['money'])/2) # divide max to 2; invert scale so that higher is better off
    #Counts columns
    if all(col in df.columns for col in ['life_meditation','life_stress','life_activity_eating','life_red_stop_alcoh','insurance','money_assistance','birth','med_stop','nonmed_stop','nonmed_start',
                                         'money_scaled']):
        X["pos_charac_sum_neg"] = -1 * df[['life_red_stop_alcoh','insurance','money_assistance','birth','med_stop','nonmed_stop','money_scaled']].sum(axis=1).astype('float32')
        X["pos_charac_sum"] = df[['life_meditation','life_stress','life_activity_eating',
                              'life_red_stop_alcoh','insurance','money_assistance','birth',
                              'med_stop','nonmed_start','money_scaled']].sum(axis=1).astype('float32')
    if all(col in df.columns for col in ['pregnant','trauma','num_migraine_days','meds_migraine','med_start','nonmed_start','med_dose','comorbid_cancer','comorbid_diabetes_typ1',
                                         'comorbid_diabetes_typ2','comorbid_gout','comorbid_migraines','comorbid_ms','comorbid_osteoporosis','comorbid_neuropathic',
                                         'comorbid_arthritis']):
        X["neg_charac_sum"] = df[['pregnant','trauma','num_migraine_days','meds_migraine',
                              'med_start','nonmed_start','med_dose','comorbid_cancer','comorbid_diabetes_typ1',
                              'comorbid_diabetes_typ2','comorbid_gout','comorbid_migraines',
                              'comorbid_ms','comorbid_osteoporosis','comorbid_neuropathic',
                              'comorbid_arthritis']].sum(axis=1).astype('float32')
        
        X["neg_charac_sum^2"] = (X["neg_charac_sum"]**2).astype('float32')
        
        if 'bmi' in df.columns:
            X["bmi^2_neg_charac_sum"] = ((df["bmi"]**2) * X["neg_charac_sum"]).astype('float32')
            X["bmi_neg_charac_sum"] = (df["bmi"]*X["neg_charac_sum"]).astype('float32')
            X["bmi_neg_charac_sum^2"] = (df["bmi"]*(X["neg_charac_sum"]**2)).astype('float32')
        if 'money' in df.columns:
            X["money^2_neg_charac_sum"] = ((df["money"]**2) * X["neg_charac_sum"]).astype('float32')
            X["money_neg_charac_sum"] = (df["money"]*X["neg_charac_sum"]).astype('float32')
            X["money_neg_charac_sum^2"] = (df["money"]*(X["neg_charac_sum"]**2)).astype('float32')
    
        if 'insurance' in df.columns:
            X["insurance^2_neg_charac_sum"] = ((df["insurance"]**2) * X["neg_charac_sum"]).astype('float32')
            X["insurance_neg_charac_sum"] = (df["insurance"]*X["neg_charac_sum"]).astype('float32')
            X["insurance_neg_charac_sum^2"] = (df["insurance"]*(X["neg_charac_sum"]**2)).astype('float32')
        
        
    if all(col in df.columns for col in ['comorbid_diabetes_typ1','comorbid_diabetes_typ2']):
        X["diabetes"] = df[['comorbid_diabetes_typ1','comorbid_diabetes_typ2']].sum(axis=1).astype('float32')

    #Intermediate column deletion
    df.drop('money_scaled',axis=1,inplace=True)
    return X

def group_transforms(df):
    """
    Performs feature engineering transformations through groupby on the input DataFrame to create new features.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the original features.
    
    Returns:
        pandas.DataFrame: The DataFrame with new count-based features.
    """
    X = pd.DataFrame()
    #Intermediate column creation
    if all(col in df.columns for col in ['race_white','race_black','race_hispanic','race_asian','race_other']):
        df['race'] = (df[['race_white','race_black','race_hispanic','race_asian','race_other']]).astype(int).idxmax(1)#create 1 column  for 'race'
    #Transforms
    #Median/mean of some numerical columns by sex ,race, education level, having insurance or receiving financial aid
    X["mean_money_sex"] = df.groupby("sex")["money"].transform("mean").astype('float32')
    X["median_bmi_sex"] = df.groupby("sex")["bmi"].transform("median").astype('float32')
    X["mean_money_race"] = df.groupby("race")["money"].transform("mean").astype('float32')
    X["median_bmi_race"] = df.groupby("race")["bmi"].transform("median").astype('float32')
    X["mean_money_educ"] = df.groupby("educ")["money"].transform("mean").astype('float32')
    X["median_bmi_educ"] = df.groupby("educ")["bmi"].transform("median").astype('float32')
    X["mean_money_insurance"] = df.groupby("insurance")["money"].transform("mean").astype('float32')
    X["median_bmi_insurance"] = df.groupby("insurance")["bmi"].transform("median").astype('float32')
    X["mean_money_money_assistance"] = df.groupby("money_assistance")["money"].transform("mean").astype('float32')
    X["median_bmi_money_assistance"] = df.groupby("money_assistance")["bmi"].transform("median").astype('float32')
    
    
    if "sleep_asleep_week_mean" in df.columns:
        X["median_steps_sex"] = df.groupby("sex")["steps_awake_mean"].transform("median").astype('float32')
        X["median_sleep_sex"] = df.groupby("sex")["sleep_asleep_week_mean"].transform("median").astype('float32')
        X["median_steps_race"] = df.groupby("race")["steps_awake_mean"].transform("median").astype('float32')
        X["median_sleep_race"] = df.groupby("race")["sleep_asleep_week_mean"].transform("median").astype('float32')
    #Intermediate column deletion
    df.drop("race",axis=1,inplace=True)
    #Return newly created columns in a DF
    return X

def create_features(df, df_test=None):
    """
    Creates new features based on the input DataFrame and performs various transformations.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the original features.
        df_test (pandas.DataFrame, optional): The DataFrame containing test data for prediction. Default is None.
    
    Returns:
        pandas.DataFrame: The DataFrame with new features.
        
    Notes:
        - If `df_test` is provided, the function combines the data from `df` and `df_test` before creating features.
        - The function applies mathematical_transforms(), counts(), and group_transforms() to create new features.
        - It also updates the sum of negative characteristics with thresholds.
        - Certain columns are dropped during the process.
    """
    X = df.copy()
    #y = X.pop("SalePrice")
    

    # Combine splits if test data is given
    #
    # If we're creating features for test set predictions, we should
    # use all the data we have available. After creating our features,
    # we'll recreate the splits.
    if df_test is not None:
        X_test = df_test.copy()
        #X_test.pop("phq9_cat_end")
        X = pd.concat([X, X_test], axis=0)
    #Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(counts(X))
    X = X.join(group_transforms(X))
    
    #Update sum of negative characteristics with thresholds
    
    if  all(col in X.columns for col in ["neg_charac_sum","sleep_threshold","steps_threshold","in_bed_threshold"]):
        X["neg_charac_sum"] = X["neg_charac_sum"] + X["sleep_threshold"] + X["steps_threshold"] + X["in_bed_threshold"]
    
    
    # #Mutual Information
    # if drop_mi:
    #     mi_scores = make_mi_scores(X, target_train)
    #     X = drop_uninformative(X, mi_scores)
    
    #Columns deletion
    try:
        X.drop(['sleep_asleep_weekday_mean','sleep_in_bed_weekday_mean',
                'sleep_asleep_weekend_mean','sleep_in_bed_weekend_mean'],axis=1)
    except:
        pass
    
    #X.drop(iqr_cols+reg_cols,axis=1)
    
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)
        
        
    if df_test is not None:
        return X, X_test
    else:
        return X