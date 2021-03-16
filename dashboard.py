#%%

import pandas as pd
import fileio
import os
import altair as alt
from typing import Tuple

alt.renderers.enable('altair_viewer')
ONE_DIR_UP = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
model = fileio.pklOpener(THIS_FOLDER + "/XGBmodel.pkl")
AVERAGE_VAL_DICT = fileio.pklOpener(THIS_FOLDER + "/averageFeatureValues.pkl")


presented_files = ['p009573.psv', 'p115835.psv', 'p111537.psv', 'p118804.psv']

def df_instantiator_unaugmented(filename: str) -> pd.DataFrame:
    """
    Takes in a filename that belongs to training_setA or training_setB and
    returns a dataframe object

    Args:
        filename: a string filename
    
    Returns:
        a dataframe object that contains the data retrieved from that filename
    """
    if filename[1] == '0':
        df = pd.read_csv(THIS_FOLDER + f'/training_setA/{filename}', sep='|')
    elif filename[1] == '1':
        df = pd.read_csv(THIS_FOLDER + f'/training_setB/{filename}', sep='|')
    return df

def df_instantiator_augmented(x: Tuple[str]) -> pd.DataFrame:
    """Creates a dataframe given filename and location.

    Args:
        x: a tuple where the first element is the directory and the second 
        element is the filename
    
    Returns:
        A dataframe created from the patient record stored under that filename
    """
    dframe = pd.read_csv(THIS_FOLDER + f'/{x[0]}/{x[1]}', 
                        sep='|')
    return dframe

def t_sepsis_col_adder(filename: str) -> pd.DataFrame:
    """Adds a column that gives the time till t_sepsis at each time step.

    Args:
        filename: name of the file being processed

    Returns:
        Returns a dataframe with the new column
    """
    df = df_instantiator_unaugmented(filename)
    number_of_sepsis_hours = df['SepsisLabel'].sum()
    if number_of_sepsis_hours > 0:
        if number_of_sepsis_hours >= 7:
            first_occurrence = df['SepsisLabel'].idxmax()
            t_sepsis = first_occurrence + 6
        else:
            t_sepsis = len(df.index) - 1
        
        temp_list = []
        for i in range(len(df.index)):
            temp_list.append(t_sepsis - i)
        df['T_Sepsis'] = temp_list
    else:
        t_end_recording = len(df.index)
        temp_list = []
        for i in range(len(df.index)):
            temp_list.append(t_end_recording - i - 1)
        df['T_EndRecording'] = temp_list
    return df

def preprocessor(filename: str) -> pd.DataFrame:
    """
    Creates and processes the dataframe for the passed in filename so 
    it contains the feature engineered features as well as processes NaN 
    values.

    Args:
        filename: name of the current patient file being processed
    
    Returns:
        the processed dataframe
    """
    HIGH = 8760
    df = df_instantiator_unaugmented(filename)
    df_null = df.isnull()
    columns = list(df.columns)[:-7]
    for column in columns:
        column_name = f"{column}_Flag"
        new_column = []
        initial_value = df_null[column][0]
        i = 0
        while initial_value == True:
            new_column.append(HIGH)
            i += 1
            if i >= len(df.index):
                break
            else:
                initial_value = df_null[column][i]
        hours_since = 0
        for j in range(i, len(df.index)):
            if df_null[column][j] == False:
                hours_since = 0
                new_column.append(hours_since)
            else:
                hours_since += 1
                new_column.append(hours_since)
        df[column_name] = new_column
    df.fillna(method='ffill', inplace=True)
    df.fillna(value=AVERAGE_VAL_DICT, inplace=True)
    
    artifact = []
    for i in range(len(df.index)):
        artifact.append(i)
    df.insert(loc=0, column='Unnamed: 0', value=artifact)
    return df

def scatter_plotter(filename: str) -> None:
    """Takes in a file name and creates the scatter plot required for that file

    Args:
        filename: name of the given file
    """
    df = t_sepsis_col_adder(filename)
    if df['SepsisLabel'].sum() > 0:
        cols = list(df.columns)
        cols_to_remove = ['Age', 'Gender', 'Unit1', 'Unit2', 'T_Sepsis', 'SepsisLabel', 'HospAdmTime']
        for col in cols:
            if df[col].isnull().all():
                cols_to_remove.append(col)
        for col in cols_to_remove:
            cols.remove(col)
        col_l, col_r = cols[:len(cols) // 2], cols[len(cols) // 2:]
        chart_l = alt.Chart(df).mark_line(point=True).encode(
            alt.X(alt.repeat("column"), type='quantitative', 
                  sort="descending", title="Time Till Sepsis"),
            alt.Y(alt.repeat("row"), type='quantitative', 
                  scale=alt.Scale(zero=False)),
            order="T_Sepsis"
        ).properties(
            width=600,
            height=100
        ).repeat(
            row=col_l,
            column=['T_Sepsis']
        )
        chart_l.save("chart_l.png")
        
        chart_r = alt.Chart(df).mark_line(point=True).encode(
            alt.X(alt.repeat("column"), type='quantitative', 
                  sort="descending", title="Time Till Sepsis"),
            alt.Y(alt.repeat("row"), type='quantitative', 
                  scale=alt.Scale(zero=False)),
            order="T_Sepsis"
        ).properties(
            width=600,
            height=100
        ).repeat(
            row=col_r,
            column=['T_Sepsis']
        )
        chart_r.save("chart_r.png")
        chart_concat = alt.hconcat(chart_l, chart_r)
        chart_concat.save("chart_concat.png")
        
        processed_df = preprocessor(filename)
        # Y_pre = processed_df['SepsisLabel']
        processed_df.drop(['SepsisLabel'], axis=1, inplace=True)
        X = processed_df.to_numpy()
        # Y = Y_pre.to_numpy().reshape(Y_pre.shape[0], 1)
        out = model.predict_proba(X)
        print(out)
    else:
        pass
        
# df = t_sepsis_col_adder('p009573.psv')
# print(df[['Temp', 'HR']])
scatter_plotter('p009573.psv')
# %%
