import pandas as pd

def tweak_df():
    """
    Reads the input parquet file and applies several transformations to it.
    
    Returns:
        pandas.DataFrame: The transformed dataframe.
    """
    df = pd.read_parquet('../data/raw/anon_processed_df.parquet')
    
    # create index to drop
    idx_to_drop = set()
    idx_to_drop |= set(df.loc[df["weight"] < 80].index.tolist()) #The |= operator performs a bitwise OR comparison and assigns the result to the left operand
    idx_to_drop |= set(df.loc[df["sleep_asleep_weekday_mean"] == 0].index.tolist())
    idx_to_drop |= set(df.loc[df["sleep_asleep_weekend_mean"] == 0].index.tolist())
    idx_to_drop |= set(df.loc[df["sleep_in_bed_weekday_mean"] == 0].index.tolist())
    idx_to_drop |= set(df.loc[df["sleep_in_bed_weekend_mean"] == 0].index.tolist())
    
    score_cols = [col for col in df.columns if '__score' in col]
    
    # drop unnecessary columns and reset index
    df = df.drop(idx_to_drop, axis=0).reset_index().rename(columns={'index': 'id_month'})
    
    # apply transformations
    df = df.assign(id=df["id_month"].str.split("_", expand=True)[0].astype('int32'),
             month=df["id_month"].str.split("_", expand=True)[1].astype('int32'),
             age= 2018 - df['birthyear'],
             sleep_asleep_week_mean=(df["sleep_asleep_weekday_mean"] * (5 / 7)) + (df["sleep_asleep_weekend_mean"] * (2 / 7)),
             sleep_in_bed_week_mean=(df["sleep_in_bed_weekday_mean"] * (5 / 7)) + (df["sleep_in_bed_weekend_mean"] * (2 / 7)))
    
    # drop unnecessary columns
    df = df.drop(["birthyear", "sleep_asleep_weekday_mean", "sleep_asleep_weekend_mean", "sleep_in_bed_weekday_mean",
         "sleep_in_bed_weekend_mean"] + score_cols, axis=1)
    
    return df

def exists(df_input, month, id, cat):
    """
    Checks whether a given category has any non-null value for a given month and ID in the input dataframe.
    
    Args:
        month (int): The month to check.
        id (int): The ID to check.
        cat (str): The category to check.
    
    Returns:
        bool: True if the category has any non-null value, False otherwise.
    """
    return df_input[(df_input['month'] == month) & (df_input['id'] == id)][cat].notna().any()

# Get the index if measurement exists at start of months 3, 6 and 9 and we have the LMC answer as well
def check_conditions(df_input, mo1, mo2, id, idx):
    """
    Checks several conditions and adds the index of the matching row to a set if they are all met.
    
    Args:
        df_input (DataFrame): Input DataFrame
        mo1 (int): The first month to check.
        mo2 (int): The second month to check.
        id (int): The ID to check.
        idx (set): A set of indices to add the matching index to.
    """
    if exists(df_input, mo1, id, "phq9_cat_start") and exists(df_input, mo2, id, "phq9_cat_start") and exists(df_input, mo2, id, "med_stop"):
        index = df_input[(df_input['month'] == mo2) & (df_input['id'] == id)].index[0]
        idx.add(index)

# Same but for start and end of months 3 or 12
def check_conditions_2(df_input, mo1, mo2, id, idx):
    """
    Checks several conditions and adds the index of the matching row to a set if they are all met.
    
    Args:
        mo1 (int): The first month to check.
        mo2 (int): The second month to check.
        id (int): The ID to check.
        idx (set): A set of indices to add the matching index to.
    """
    if exists(df_input, mo1, id, "phq9_cat_start") and exists(df_input, mo1, id,"phq9_cat_end") and exists(df_input, mo2, id, "med_stop"):
        index = df_input[(df_input['month'] == mo2) & (df_input['id'] == id)].index[0]
        idx.add(index)
