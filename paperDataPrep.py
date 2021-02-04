import fileio
from typing import Dict, List, Tuple
import recordRetrieval
import pandas as pd
import random

def df_instantiator(x: Tuple[str]) -> pd.DataFrame:
    """Creates a dataframe given filename and location.

    Args:
        x: a tuple where the first element is the directory and the second 
        element is the filename
    
    Returns:
        A dataframe created from the patient record stored under that filename
    """
    dframe = pd.read_csv(recordRetrieval.THIS_FOLDER + f'/{x[0]}/{x[1]}', 
                        sep='|')
    return dframe

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

