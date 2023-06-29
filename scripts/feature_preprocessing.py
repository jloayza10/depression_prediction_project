import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PowerTransformer, FunctionTransformer, KBinsDiscretizer, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
import sys
sys.path.append('../scripts')
from preprocessing import load_list_from_pkl


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

def to_float32(df):
    return pd.DataFrame(df).astype('float32')

def freq_encode(df):
    freq = df.value_counts(normalize=True)
    return pd.DataFrame(df.apply(lambda x : freq[x]))

def get_preprocessor(df):
    """
    Returns a preprocessor pipeline for transforming the input DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to be preprocessed.

    Returns:
        sklearn.compose.ColumnTransformer: The preprocessor pipeline.

    Notes:
        - The function defines various column lists based on the input DataFrame.
        - It also defines imputers, scalers, transformers, and pipelines for different preprocessing steps.
        - The preprocessor pipeline combines these transformers and applies them to the DataFrame.
    """
    list_path = '../data/lists/'
    impute_most_f_scale_cols = load_list_from_pkl(list_path+'impute_most_f_scale_cols.pkl')
    impute_median_scale_cols = load_list_from_pkl(list_path+'impute_median_scale_cols.pkl')
    impute_most_f_yeo_cols = load_list_from_pkl(list_path+'impute_most_f_yeo_cols.pkl')
    impute_median_yeo_cols = load_list_from_pkl(list_path+'impute_median_yeo_cols.pkl')
    impute_most_f_scale_yeo_cols = load_list_from_pkl(list_path+'impute_most_f_scale_yeo_cols.pkl')
    impute_median_scale_yeo_cols = load_list_from_pkl(list_path+'impute_median_scale_yeo_cols.pkl')
    scale_yeo_cols = load_list_from_pkl(list_path+'scale_yeo_cols.pkl')
    impute_most_f_only = load_list_from_pkl(list_path+'impute_most_f_only.pkl')
    impute_median_only = load_list_from_pkl(list_path+'impute_median_only.pkl')
    scale_cols_only = load_list_from_pkl(list_path+'scale_cols_only.pkl')
    encode_cols_only = load_list_from_pkl(list_path+'encode_cols_only.pkl')
    to_bin = load_list_from_pkl(list_path+'to_bin.pkl')
    to_encode = load_list_from_pkl(list_path+'to_encode.pkl')
    
    #Column lists function definitions
    yeo_most_f_cols = [col for col in df.columns if col in impute_most_f_scale_yeo_cols]
    yeo_median_cols = [col for col in df.columns if col in impute_median_scale_yeo_cols]
    yeo_impute_most_f_cols = [col for col in df.columns if col in impute_most_f_yeo_cols]
    yeo_impute_median_cols = [col for col in df.columns if col in impute_median_yeo_cols]
    yeo_cols =  [col for col in df.columns if col in scale_yeo_cols]
    impute_most_f_only_cols = [col for col in df.columns if col in impute_most_f_only]
    impute_median_only_cols = [col for col in df.columns if col in impute_median_only]
    most_f_cols = [col for col in df.columns if col in impute_most_f_scale_cols]
    median_cols = [col for col in df.columns if col in impute_median_scale_cols]
    scaled_cols = [col for col in df.columns if col in scale_cols_only]
    binned_cols = [col for col in df.columns if col in to_bin]
    encode_cols = [col for col in df.columns if col in to_encode]
    remainder_cols = list(set(df.columns) - set(yeo_most_f_cols) - set(yeo_median_cols) - set(yeo_impute_most_f_cols) - set(yeo_impute_median_cols) - set(yeo_cols) 
                          - set(impute_most_f_only_cols) - set(impute_median_only_cols) - set(most_f_cols) - set(median_cols) - set(scaled_cols) - set(binned_cols)
                         - set(encode_cols))
    
    
    #Imputer definition
    imputer_most_f = SimpleImputer(strategy='most_frequent')
    imputer_median = SimpleImputer(strategy='median')
    imputer_mean = SimpleImputer(strategy='mean')
    #Scaling
    scaler = StandardScaler()
    #Normalize
    power_yeo = PowerTransformer(method='yeo-johnson',standardize=False) # if standardize, then no need for scaling
    # Binning
    binning = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    # One-hot encoding
    encoder = OneHotEncoder(handle_unknown='ignore') #drop = 'first',
    
    #function for categorical to integer
    # def to_float32(df):
    #     return pd.DataFrame(df).astype('float32')
          
    
    float32_transformer = FunctionTransformer(to_float32)
    freq_transformer = FunctionTransformer(freq_encode)
    
    
    #Pipeline definition
    
    imputer_most_f_to_float_pipe = Pipeline(steps=[
        ('imputer_most_f', imputer_most_f),
        ('to_float32', float32_transformer)        
    ])
    imputer_median_to_float_pipe = Pipeline(steps=[
        ('imputer_median', imputer_median),
        ('to_float32', float32_transformer)        
    ])
    imputer_most_f_scale_yeo_pipe = Pipeline(steps=[
        ('imputer_most_f', imputer_most_f),
        ('scale', scaler),
        ('yeo', power_yeo)
    ])
    imputer_median_scale_yeo_pipe = Pipeline(steps=[
        ('imputer_median', imputer_median),
        ('scale', scaler),
        ('yeo', power_yeo)
    ])
    imputer_most_f_yeo_pipe = Pipeline(steps=[
        ('imputer_most_f', imputer_most_f),
        ('yeo', power_yeo)
    ])
    imputer_median_yeo_pipe = Pipeline(steps=[
        ('imputer_median', imputer_median),
        ('yeo', power_yeo)
    ])
    
    imputer_most_f_scale_pipe = Pipeline(steps=[
        ('imputer_most_f', imputer_most_f),
        ('scale', scaler)        
    ])
    imputer_median_scale_pipe = Pipeline(steps=[
        ('imputer_median', imputer_median),
        ('scale', scaler)        
    ])
       
    scale_yeo_pipe = Pipeline(steps=[
        ('scale', scaler),
        ('yeo', power_yeo)
    ])
    
    binning_pipe = Pipeline(steps=[
        ('imputer_most_f', imputer_most_f),
        ('binning', binning),
        #('onehot', encoder)
    ])
    encoder_pipe = Pipeline(steps=[
        ('imputer_most_f', imputer_most_f),
        ('onehot', encoder)
    ])
     # imputer_mean_scale_pipe = Pipeline(steps=[
    #     ('imputer_mean', imputer_mean),
    #     ('scale', scaler)        
    # ])
    # encoder_pipe = Pipeline(steps=[
    #     ('imputer_most_f', imputer_most_f),
    #     ('tgt_encoder', category_encoders.TargetEncoder())
    # ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('imputer_most_f_to_float', imputer_most_f_to_float_pipe, impute_most_f_only_cols),
            ('imputer_median_to_float', imputer_median_to_float_pipe, impute_median_only_cols),
            ('imputer_most_f_scale', imputer_most_f_scale_pipe, most_f_cols),
            ('imputer_median_scale', imputer_median_scale_pipe, median_cols),
            ('imputer_most_f_scale_yeo', imputer_most_f_scale_yeo_pipe, yeo_most_f_cols),
            ('imputer_median_scale_yeo', imputer_median_scale_yeo_pipe, yeo_median_cols),
            ('imputer_most_f_yeo', imputer_most_f_yeo_pipe, yeo_impute_most_f_cols),
            ('imputer_median_yeo', imputer_median_yeo_pipe, yeo_impute_median_cols),            
            ('scale_yeo_pipe', scale_yeo_pipe, yeo_cols),
            ('scaler', scaler, scaled_cols),
            ('binning', binning_pipe, binned_cols),
            ('encoder', encoder_pipe, encode_cols),
            ('float32_transformer', float32_transformer, remainder_cols)
            #('imputer_mean', imputer_mean_scale_pipe, mean_cols),
            #('tgt_encoder', encoder_pipe, encode_cols),
            #('freq_encoder', freq_transformer, encode_cols),
        ],
        remainder="drop" # the remaining columns not transformed are dropped
    )
    return preprocessor

def preprocess_dataframe(df, df_test=None):
    """
    Preprocesses a DataFrame using a given preprocessor.

    Args:
        df (pd.DataFrame): Input DataFrame to be preprocessed.
        df_test (pd.DataFrame): Input Test DataFrame to be preprocessed if needed.
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame(s).
    """
    
    preprocessor = get_preprocessor(df)
    # Apply the preprocessor to the DataFrame
    df_preprocessed = preprocessor.fit_transform(df)

    # Convert the preprocessed data to a DataFrame with column names
    df_preprocessed = pd.DataFrame(df_preprocessed, columns=get_feature_names(preprocessor))
    
    if df_test is not None:
        df_test_preprocessed = preprocessor.transform(df_test)
        df_test_preprocessed = pd.DataFrame(df_test_preprocessed, columns=get_feature_names(preprocessor))
        return df_preprocessed, df_test_preprocessed
    else:
        return df_preprocessed
    
