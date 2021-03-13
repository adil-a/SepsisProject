import fileio
from typing import Dict, List, Tuple
import recordRetrieval
import pandas as pd
import random
import MeanMedianIQR
import os

AVERAGE_VAL_DICT = fileio.pklOpener("averageFeatureValues.pkl")
ONE_DIR_UP = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

# no longer used
def df_instantiator(x: Tuple[str]) -> pd.DataFrame:
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

# no longer used
def file_to_df() -> None:
    """Uses the recordRetrieval.retrieveAllFiles() method to convert the 
    list of file names in the value pair to a list of data frames instead. 
    This new dictionary is then saved.
    
    Args:
        None
    
    Returns:
        None
    """
    filenamesdict = recordRetrieval.retrieveAllFilesFolders(recordRetrieval.retrieveAllFiles())
    average_values = fileio.pklOpener("averageFeatureValues.pkl")
    dictionary = {}

    for directory in filenamesdict:
        test_frames = []
        for filename in filenamesdict[directory]:
            temp_df = df_instantiator((directory, filename))
            temp_df['Filename'] = [filename] * len(temp_df.index)
            test_frames.append(temp_df)
        testDF = pd.concat(test_frames)
        testDF.ffill(inplace=True)
        testDF.fillna(value=average_values, inplace=True)
        dictionary[directory] = testDF

    fileio.StraightDumpDir(dictionary, "dictWithDirAndDF.pkl")

# no longer used
def new_sets() -> None:
    """Adds on seven (randomly chosen) of the newly introduced sets to the 
    original training set DataFrame.

    Args:
        None

    Returns:
        None
    """
    dict_with_DFs = fileio.pklOpener("dictWithDirAndDF.pkl")
    new_set = [fileio.pklOpener("trainSetDForiginal.pkl")]
    lst = list(dict_with_DFs)
    for _ in range(7):
        choice = random.choice(lst)
        lst.remove(choice)
        new_set.append(dict_with_DFs[choice])
    print(lst)
    new_DF = pd.concat(new_set)
    fileio.StraightDumpDir(new_DF, "newTrainSet.pkl")

def flag_adder() -> None:
    """
    Augments each psv file so that we replace the NaN values using 
    fill-forward and the remaining NaN values are replaced with average 
    values calculated at an earlier point. Additionally, it adds a flag 
    column for each of the non-constant features (34 of the 40 features are 
    non-constant). This flag column is used to find the last time data was 
    recorded for the respective feature. For example:
    Temp Temp_Flag
    NaN     8760
    NaN     8760
    36.3    0
    NaN     1
    37      0
    NaN     1
    NaN     2
    The reason the first two rows of Temp_Flag are so high is because there 
    are is no previously recorded data for Temp initially, so we set this to 
    be the number of hours in a year by default (since no patient record 
    goes that long).

    Args:
        None
    
    Returns:
        None
    """
    training_filenames = fileio.pklOpener("trainSetList.pkl")
    test_filenames = fileio.pklOpener("testSetList.pkl")
    for filename in training_filenames:
        flag_adder_helper(filename, "train")
    for filename in test_filenames:
        flag_adder_helper(filename, "test")
        
def flag_adder_helper(filename: str, folder_type: str) -> None:
    """
    Helper for the function flag_adder. Used to reduce redundant code. This 
    helper and the main flag_adder function together complete the augmentation 
    described in the docstring for flag_adder.

    Args:
        filename: name of the current patient file being processed
        folder_type: the function stores in the correct directory depending on 
            if we are currently processing training files or test files.
    
    Returns:
        None
    """
    HIGH = 8760
    df = MeanMedianIQR.df_instantiator(filename)
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
    if folder_type == "train":
        df.to_csv(THIS_FOLDER + f"/trainingSetAugmented/{filename}", sep="|")
    elif folder_type == "test":
        df.to_csv(THIS_FOLDER + f"/testSetAugmented/{filename}", sep="|")

def train_test_df_creator() -> None:
    """
    Creates and stores the train/test dataframes from the augmented dataset

    Args:
        None
    
    Returns:
        None
    """
    train_set_list = os.listdir(THIS_FOLDER + "/trainingSetAugmented")
    test_set_list = os.listdir(THIS_FOLDER + "/testSetAugmented")
    train_frames_1 = []
    train_frames_2 = []
    test_frames = []
    for i in range(0, len(train_set_list) // 2):
        df = df_instantiator(("trainingSetAugmented", train_set_list[i]))
        train_frames_1.append(df)
    for i in range(len(train_set_list) // 2, len(train_set_list)):
        df = df_instantiator(("trainingSetAugmented", train_set_list[i]))
        train_frames_2.append(df)
    for filename in test_set_list:
        df = df_instantiator(("testSetAugmented", filename))
        test_frames.append(df)
    trainSetDF_batch1 = pd.concat(train_frames_1)
    trainSetDF_batch2 = pd.concat(train_frames_2)
    testSetDF = pd.concat(test_frames)
    fileio.StraightDumpDir(trainSetDF_batch1, THIS_FOLDER + "/trainingSetAugmentedDF_batch1.pkl")
    fileio.StraightDumpDir(trainSetDF_batch2, THIS_FOLDER + "/trainingSetAugmentedDF_batch2.pkl")
    fileio.StraightDumpDir(testSetDF, THIS_FOLDER + "/testSetAugmentedDF.pkl")

# train_test_df_creator()