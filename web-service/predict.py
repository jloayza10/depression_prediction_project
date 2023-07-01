import pickle
import pandas as pd

with open('xgboost_mod.bin', 'rb') as f_in:
    model = pickle.load(f_in)

with open('final_preprocessor.bin', 'rb') as f_in:
    preprocessor = pickle.load(f_in)

from feature_preprocessing import preprocess_dataframe, create_features, load_list_from_pkl, get_feature_names

list_path = '../data/lists/'


def load_list_from_pkl(file_path):
    """
    Load a pickled list object from a file.

    Parameters:
        file_path (str): The path to the pickle file.

    Returns:
        list: The loaded list object.
    """
    with open(file_path, 'rb') as file:
        list_object = pickle.load(file)
    return list_object

cols_to_del = load_list_from_pkl(list_path+'cols_to_del.pkl')
selected_features_2 = load_list_from_pkl('../data/lists/selected_features_2.pkl')
print(f'Type of selected features: {type(selected_features_2)}')

def read_data(input):
    df = pd.DataFrame.from_dict([input])
    score_cols = [col for col in df.columns if '__score' in col]
    
    # drop unnecessary columns and reset index
    df = df.rename(columns={'index': 'id_month'})
    
    # apply transformations
    df = df.assign(id=df["id_month"].str.split("_", expand=True)[0].astype('int32'),
             month=df["id_month"].str.split("_", expand=True)[1].astype('int32'),
             age= 2018 - df['birthyear'],
             sleep_asleep_week_mean=(df["sleep_asleep_weekday_mean"] * (5 / 7)) + (df["sleep_asleep_weekend_mean"] * (2 / 7)),
             sleep_in_bed_week_mean=(df["sleep_in_bed_weekday_mean"] * (5 / 7)) + (df["sleep_in_bed_weekend_mean"] * (2 / 7)))
    
    # drop unnecessary columns
    df = df.drop(["birthyear", "sleep_asleep_weekday_mean", "sleep_asleep_weekend_mean", "sleep_in_bed_weekday_mean",
         "sleep_in_bed_weekend_mean"] + score_cols, axis=1)
    
    print(f'Columns have been dropped, starting create features')
    df = create_features(df)
    print(f'Starting preprocessing')
    df = preprocessor.transform(df)
    print(f'Finished preprocessing')
    df_preprocessed = pd.DataFrame(df, columns=get_feature_names(preprocessor))[selected_features_2]
    
    return df_preprocessed

def predict(df):
    print(f'----------------Starting prediction-----------------')
    preds = model.predict(df)
    print(f'-----------------Finished prediction-----------------')
    return float(preds)