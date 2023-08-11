import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

fig_path = '../figures/'
def get_missing_cols(df, lower_threshold, upper_threshold):
    """
    Retrieves columns from a DataFrame that have a percentage of missing values within a specified threshold.
    
    Args:
        df (pandas.DataFrame): The DataFrame to check for missing values.
        lower_threshold (float): The lower threshold for the percentage of missing values (exclusive).
        upper_threshold (float): The upper threshold for the percentage of missing values (exclusive).
    
    Returns:
        list: A list of column names that meet the specified threshold criteria.
    """
    cols_missing = []
    total_rows = df.shape[0]
    for col in df.columns:
        missing_values = df[col].isnull().sum()
        percent_missing = missing_values / total_rows
        if percent_missing > lower_threshold and percent_missing < upper_threshold:
            print(f"{col} has {missing_values} missing values ({percent_missing:.2%} missing)")
            cols_missing.append(col)
    return cols_missing

def plot_target_dist(df, save=False):
    plt.figure(figsize=(18, 8))
    plt.hist(df["phq9_cat_end"], align='mid', bins=[-0.25, 0.25, 0.75,1.25, 1.75,2.25,2.75,3.25,3.75,4.25])
    plt.xlabel("Self-reported depression value")
    plt.ylabel("Count")
    plt.title("Target Distribution (Depression Severity)", size =20, y = 1.05)
    labels = ['0:\n low severity', '1', '2', '3', '4:\n high severity']
    plt.xticks([0, 1, 2, 3, 4], labels)    
    if save:
        fig_title = 'target_distribution'
        plt.savefig(fig_path+fig_title)
    plt.show()


def plot_histograms(df, cols, size):
    """
    Plots histograms for specified columns in a DataFrame.
    
    Args:
        df (pandas.DataFrame): The DataFrame containing the columns to plot.
        cols (list): A list of column names to plot histograms for.
        size (tuple): The size of the resulting figure (width, height).
    """
    n_cols = min(len(cols), 5)
    n_rows = - (-len(cols) // n_cols)
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=size)
    axs = axs.flatten()
    for i, col in enumerate(cols):
        axs[i].set_xlabel(col, fontsize=16)
        axs[i].set_ylabel('Count', fontsize=16)
        if df[col].dtype == bool:
            df[col].dropna().astype(int).hist(ax=axs[i])
        elif df[col].dtype == object:
            pass
        else:
            df[col].dropna().hist(ax=axs[i])
    plt.show()
    
def plot_percent_by_sex(df, save=False):
    #title_fig = '.jpg'
    group_by_df = (df.groupby('sex')["phq9_cat_end"]
                   .value_counts(normalize=True)
                   .mul(100)
                   .rename('percent')
                   .reset_index())
    plt.figure(figsize=(14,8))
    g = sns.histplot(x = 'sex', 
                     hue = "phq9_cat_end",
                     weights= 'percent',
                     #hue_order = ['4.0','3.0','2.0','1.0','0.0',],
                     #palette=[sns.color_palette()[2],sns.color_palette()[0],sns.color_palette()[1]],
                     multiple = 'stack',
                     data= group_by_df,
                     shrink = 0.5,
                     discrete=True,
                     legend=True)
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(0, 101, 20), np.arange(100, -1, -20))
    plt.xlabel('Sex', fontsize=18)
    g.set(xticks=range(0,3,1),xticklabels=["Female","Male","Other"])
    plt.ylabel('Percentage', fontsize=18)
    plt.title("Percentual distribution of depression severity by gender", y=1.05, size = 20)
    sns.move_legend(g, loc = "center left",labels=['4 (high severity)','3','2','1','0 (low severity)'],title="Depression Severity", bbox_to_anchor=(1, .51))

    for rect in g.patches:

        h = rect.get_height()
        w = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        if h>0:
            g.annotate(f'{float(h):.0f}%', xy=(x + w/2,y +h/2), 
                       xytext=(0, 0), textcoords='offset points', ha='center', va='center'
                      )
    if save:
        title_fig = 'percent_target_distribution_by_sex'
        g.figure.savefig(fig_path+title_fig,bbox_inches='tight')

def plot_dist_by_sex(df, save=False):
    g = sns.displot(df.sort_values("phq9_cat_end",ascending=True).reset_index(),
                    x="sex", 
                    hue="phq9_cat_end", 
                    multiple="dodge", 
                    height=6, 
                    aspect=1.5)
    ax = g.facet_axis
    sns.move_legend(g, loc = "center left",labels=['4 (high severity)','3','2','1','0 (low severity)'],title="Depression Severity", bbox_to_anchor=(.8, .51))
    g.set(xticks=range(0,3,1),xticklabels=["Female","Male","Other"])
    g.fig.suptitle('Distribution of depression severity by gender', fontsize=20, y=1.05)
    if save:
        title_fig = 'target_distribution_by_sex'
        g.figure.savefig(fig_path+title_fig,bbox_inches='tight')

def plot_dist_by_money(df, save = False):
    g = sns.displot(df.sort_values("phq9_cat_end",ascending=True).reset_index(),
                    x="money", 
                    hue="phq9_cat_end", 
                    multiple="dodge", 
                    height=7, 
                    aspect=1.75)
    #ax = g.facet_axis
    sns.move_legend(g, loc = "center left",labels=['4 (high severity)','3','2','1','0 (low severity)'],title="Depression Severity", bbox_to_anchor=(0.9, .51))
    g.set(xticks=range(0,5,1), xticklabels=["0: Better off","1","2","3","4: Worse off"])
    g.fig.suptitle('Distribution of depression severity by economic situation', fontsize=20, y=1.05)
    plt.xticks(fontsize=18)
    plt.xlabel('Economic situation', fontsize=20)
    plt.yticks(fontsize=18)
    plt.ylabel('Total', fontsize=20)
    if save:
        title_fig = 'target_distribution_by_economy'
        g.figure.savefig(fig_path+title_fig,bbox_inches='tight')
    
def plot_count(df,feature, save = False):
    if feature=='sex':
        normalized_df = df[feature].value_counts(normalize=True)*100
        title='Dataset gender distribution: total and by percentage'
        xticklabels=["Female","Male","Other"]
        title_fig = 'plot_count_sex'
    if feature == 'race':
        test_df = pd.DataFrame((df[['race_white','race_black','race_hispanic','race_asian','race_other']]).astype(int).idxmax(1))
        test_df = test_df.rename(columns={0: "race"})
        normalized_df = test_df.value_counts(normalize=True)*100
        title ='Dataset race distribution: total and by percentage'
        xticklabels = ["White","Black","Hispanic","Asian","Other"]
        title_fig = 'plot_count_race'
        df = test_df
        
    
    plt.figure(figsize=(9,6))
    g = sns.countplot(x=feature, data=df, order=df[feature].value_counts().index)
    g.set(xticklabels=xticklabels)
    plt.title(label = title,fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('', fontsize=20)
    plt.ylabel('Total', fontsize=20)
    for a, rect in zip(normalized_df, g.patches):
        h = rect.get_height()
        w = rect.get_width()
        x = rect.get_x()
        y = rect.get_y()
        if h>0:
            g.annotate(f'{float(a):.1f}%', xy=(x + w/2,y +h/2), xytext=(0, 0), textcoords='offset points', ha='center', va='center', fontsize=20)
    if save:
        g.figure.savefig(fig_path+title_fig,bbox_inches='tight')

def save_list_to_pkl(list_object, path, file_name):
    """
    Save a list object as a pickle file.

    Parameters:
        list_object (list): The list object to be saved.
        path (str): The directory path where the file will be saved.
        file_name (str): The desired name of the pickle file (without the extension).

    Returns:
        None
    """
    with open(path + file_name + '.pkl', 'wb') as file:
        pickle.dump(list_object, file)
        
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

def update_metadata_file(df_name, df_shape, comments, metadata_file):
    """
    Update the metadata file with information about a DataFrame.

    Parameters:
        df_name (str): The name of the DataFrame.
        df_shape (tuple): The shape of the DataFrame.
        comments (str): Any comments or additional information about the DataFrame.
        metadata_file (str): The path to the metadata file.

    Returns:
        pd.DataFrame: The updated metadata DataFrame.
    """
    # Load existing metadata file
    metadata_df = pd.read_csv(metadata_file)

    # Check if df_name already exists in the metadata DataFrame
    if df_name in metadata_df['DataFrame'].values:
        # Replace the existing row with new information
        metadata_df.loc[metadata_df['DataFrame'] == df_name, ['Shape', 'Comments']] = [df_shape, comments]
    else:
        # Create a new row with information about the DataFrame
        new_row = {
            'DataFrame': df_name,
            'Shape': df_shape,
            'Comments': comments
        }
        # Append the new row to the metadata DataFrame
        metadata_df = metadata_df.append(new_row, ignore_index=True)

    # Save the updated metadata DataFrame back to the file
    metadata_df.to_csv(metadata_file, index=False)

    # Return the updated metadata DataFrame
    return metadata_df