import pandas as pd
import fileio
import os
import seaborn as sb
from typing import Tuple

ONE_DIR_UP = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

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
    # df = df_instantiator_augmented(("testSetAugmented", filename))
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

def scatter_plotter(filename: str) -> None:
    """Takes in a file name and creates the scatter plot required for that file

    Args:
        filename: name of the given file
    """
    df = t_sepsis_col_adder(filename)
    # df.drop(['Unnamed: 0'], axis=1, inplace=True)
    # print(df)
    if df['SepsisLabel'].sum() > 0:
        cols = list(df.columns)
        cols.remove("T_Sepsis")
        g = sb.lineplot(x='T_Sepsis', y=cols, data=df)
        g.invert_xaxis()
        g.plot()
        plt = g.get_figure()
        plt.savefig("test.png")
    else:
        pass
        
# df = t_sepsis_col_adder('p009573.psv')
# print(df[['Temp', 'HR']])
scatter_plotter('p009573.psv')
# fmri = sb.load_dataset("fmri")
# print(fmri)