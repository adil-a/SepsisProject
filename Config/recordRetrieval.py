import os
from typing import List, Dict
import pandas as pd

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

def findPositiveFiles() -> List[str]:
    """
    Returns a list of file names that are positive for sepsis from 
    training_setA and training_setB.

    Args:
        None.
    
    Returns:
        A list of file names that are positive for sepsis.
    """
    lst = []
    for filename in os.listdir(THIS_FOLDER + '/training_setA'):
        df = pd.read_csv(THIS_FOLDER + f'/training_setA/{filename}', 
                    sep='|')
        total = df['SepsisLabel'].sum()
        if total > 0:
            lst.append(filename)
    for filename in os.listdir(THIS_FOLDER + '/training_setB'):
        df = pd.read_csv(THIS_FOLDER + f'/training_setB/{filename}', 
                        sep='|')
        total = df['SepsisLabel'].sum()
        if total > 0:
            lst.append(filename)
    return lst


def retrieveAllFiles() -> List[str]:
    """Retrieves all record names.

    Args:
        None
    
    Returns:
        A list of strings which contains the names of all records
    """
    folders = ['eicu_set73', 'eicu_set79', 'eicu_set167', 'eicu_set176', 
                'eicu_set199', 'eicu_set243', 'eicu_set264', 
                'eicu_set338', 'eicu_set420', 'eicu_set443', 
                'eicu_set458', 'training_setA', 'training_setB']
    return_lst = []
    for folder in folders:
        for filename in os.listdir(THIS_FOLDER + f'/{folder}'):
            return_lst.append(filename)
    return return_lst

def retrieveAllFilesFolders(filenames: List[str]) -> Dict[str, List[str]]:
    """Retrieves a dictionary where the key is the directory name and 
    the value is a list of strings representing the filenames in that 
    directory.

    Args:
        filenames: a list of all filenames in all directories
    
    Returns:
        a dictionary where the key is the directory name and the value is a 
        list of strings representing the filenames in that directory
    """
    return_dict = {}
    folders = ['eicu_set73', 'eicu_set79', 'eicu_set167', 'eicu_set176', 
                'eicu_set199', 'eicu_set243', 'eicu_set264', 
                'eicu_set338', 'eicu_set420', 'eicu_set443', 
                'eicu_set458', 'training_setA', 
                'training_setB']
    for filename in filenames:
        for directory in folders:
            if os.path.isfile(THIS_FOLDER + f'/{directory}/{filename}'):
                if directory in return_dict:
                    return_dict[directory].append(filename)
                else:
                    return_dict[directory] = [filename]
    return return_dict
