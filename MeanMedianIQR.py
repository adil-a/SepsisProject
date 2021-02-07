import os
import pandas as pd
from typing import List, Tuple, Dict, Any
import numpy as np

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

def df_instantiator(filename: str) -> pd.DataFrame:
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

def update_dict(dictionary: Dict[Any, Any], key: Any, value: Any, \
    mode: str) -> None:
    """
    Simple helper function to update dictionaries and reduce redundant code
    """
    if mode == 'accumulate':
        if key not in dictionary:
            dictionary[key] = 1
        else:
            dictionary[key] += 1
    elif mode == 'list':
        if key not in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)

def getNumPatients() -> int:
    """
    Returns the total number of patients from hospitals A and B 
    (training_setA and training_setB respectively).

    Args:
        None

    Returns:
        The number of total patients stored in directories 
        'training_setA' and 'training_setB'.
    """
    trainAdir = os.path.join(THIS_FOLDER, 'training_setA')
    trainBdir = os.path.join(THIS_FOLDER, 'training_setB')

    patientsA = os.listdir(trainAdir)
    patientsB = os.listdir(trainBdir)
    numberOfPatients = len(patientsA) + len(patientsB)
    return numberOfPatients

def findPositiveFiles() -> List[str]:
    """
    Returns a list of file names that are positive for sepsis from 
    training_setA and training_setB

    Args:
        None
    
    Returns:
        A list of file names that are positive for sepsis
    """
    lst = []
    for filename in os.listdir(THIS_FOLDER + '/training_setA'):
        df = df_instantiator(filename)
        total = df['SepsisLabel'].sum()
        if total > 0:
            lst.append(filename)
    for filename in os.listdir(THIS_FOLDER + '/training_setB'):
        df = df_instantiator(filename)
        total = df['SepsisLabel'].sum()
        if total > 0:
            lst.append(filename)
    return lst

def l10rows(lst: List[str]) -> Tuple[int, int, int, List[str]]:
    """
    Returns information on positive records with < 10 rows.
    Sepsis positive files vary in the number of rows, where each row 
    represents an hour. Under the assumption that the files contain the 1
    labels in the final column in the range t_sepsis - 6 <= t <= t_sepsis + 3, 
    we need at least 7 rows in a file so we can correctly compute the time of 
    t_sepsis, otherwise there will be some bias because we don't have enough 
    1s, which is 7. However, in this case all positive records have at least 8 
    rows and it turns out that all positive records with < 10 rows have all 1s 
    in their 'SepsisLabel' column. This function is just to be used when 
    calculating different quantities to do with time till t_sepsis.

    Args:
        lst: a sorted list of file names that are positive for sepsis.

    Returns:
        Four element tuple where the first three elements are of type int and 
        the last element is a list of strings. The first element is the total 
        number of positive records that have < 10 rows. The second element is 
        the total number records that have the same number of rows as the 
        number of 1s in their final colmn (this servers as a sanity check and 
        should be equal to the first element). The third element is to record 
        the positive record that has < 10 rows and is the smallest record in 
        that subset. The fourth element is a list of all the record names that 
        have their total number of rows equal to the number of 1s they have 
        in their final column. The length of this is equal to the second 
        element.
    """
    total = 0
    new_lst = []
    smallest_record = 15 # weknow all records that matter are < 10 rows
    for filename in lst:
        df = df_instantiator(filename)
        if len(df.index) < 10:
            total += 1
            if len(df.index) == df['SepsisLabel'].sum():
                new_lst.append(filename)
            if len(df.index) < smallest_record:
                smallest_record = len(df.index)
    return total, len(new_lst), smallest_record, new_lst

def geq10Rows(lst: List[str]) -> Tuple[int, Dict[int, int], \
    Dict[int, List[str]]]:
    """
    Returns the information on positive records with more than 10 rows. 
    There are some positive records that have > 10 rows, but have < 10 
    rows of 1 labels in the final column.

    Args:
        lst: a sorted list of file names that are positive for sepsis
    
    Returns:
        A tuple of three elemnts. Note that ever file here has at least 10 
        rows. First element denotes the number of positive records with at 
        least 10 rows. The second element is a dictionary with key:value 
        both of type int. The dictionary has keys <= 10 and teh respective 
        value pair represents the number of positive records with the number 
        of that key's 1s in their final columns. The third element is a 
        dictonary with the key being the number of 1s in the final column 
        and the value being a list of positive files that have that many 1s in 
        their final column.
        For example:
        
        (2564, {10: 1870, 9: 673, 8: 2, 6: 5, 7: 4, 4: 2, 2: 3, 1: 5}, 
        {10: ['p000009.psv', ...], 9: ['p000018.psv', ...], 
        7: ['p000639.psv', ...], 8: ['p003639.psv', ...], 
        6: ['p005639.psv', ...], 1: ['p101933.psv', ...], 
        4: ['p105030.psv', ...], 2: ['p109202.psv', ...]})

        This means that we have 2564 total positive records with >= 10 rows, 
        1870 positive records have exactly 10 1s in their final column. 
        Let's call the keys x and values y in the dictionary for exmaple's 
        sake. This means that for each key:value pair, there are y files that 
        have x number of 1s in their final column
    """
    dic1 = {}
    dic2 = {}
    total_geq_10_rows = 0
    for filename in lst:
        df = df_instantiator(filename)
        if len(df.index) >= 10:
            total_geq_10_rows += 1
            update_dict(dic1, df['SepsisLabel'].sum(), None, 'accumulate')
            update_dict(dic2, df['SepsisLabel'].sum(), filename, 'list')
    return total_geq_10_rows, dic1, dic2

def l10RowsSepsis(lst: List[str]) -> List[int]:
    """
    Fetches t_sepsis from positive records with < 10 rows.

    If the record contains 8 rows, then t_sepsis is at 8 - (8 - 7) = 7 and 
    if the record contains 9 rows, then t_sepsis is at 9 - (9 - 7) = 7. In 
    either case, t_sepsis occurs at the 7th hour after ICU admission.

    Args:
        lst: List of record names in the bias
    
    Returns:
        returns a list of 
    """
    return_lst = []
    for i in range(len(lst)):
        return_lst.append(7)
    return return_lst

def geq10RowsSepsis(dictionary: Dict[int, List[str]]) -> List[int]:
    """
    Fetches t_sepsis from positive records with >= 10 rows.

    First, we note that in all positive records, the final n consecutive rows 
    have all n 1s.

    If the record contains > 7 1s in the final column, then t_sepsis occurs at 
    len(df.index) - (number of 1s - 7). Otherwise, we take the final row as 
    t_sepsis and this is where our bias comes in as noted in the project 
    documentation.

    Args:
        dictionary: dictionary with the key being the number of 1s in the 
        final column and the value being a list of positive files that have 
        that many 1s in their final column.
    
    Returns:
        A list of integers tha tcontians the time till t_sepsis for each 
        positive record with >= 10 rows.
    """
    return_lst = []
    for key in dictionary:
        for filename in dictionary[key]:
            df = df_instantiator(filename)
            if key > 7:
                x = key - 7
                t_sepsis = len(df.index) - x
                return_lst.append(t_sepsis)
            else:
                t_sepsis = len(df.index)
                return_lst.append(t_sepsis)
    return return_lst

def sepsis_onset(l10rowslst: List[str], geq10rowsdict: Dict[int, List[str]]) -> \
    Tuple[int, int, int]:
    """
    Returns information about sepsis onset time.
    
    For our bias, wherever we have < 7 rows of 1s we will assume that the 
    final row containing the 1 label in the final column is t_sepsis. Out of 
    the total 2,932 positive records, only 10 records have < 7 rows of 1s so 
    the impact of this bias should be very minimal.

    Args:
        l10rowslst: the return list (fourth element) from the function l10rows
        geq10rowsdict: the return dict from the function geq10rows
    
    Returns:
        Returns a tuple of three elemnts where the elements are the mean, 
        median and IQR of the number of hours till t_sepsis in that order.
    """
    lst = []
    dic = {}
    lst.extend(l10RowsSepsis(l10rowslst))
    lst.extend(geq10RowsSepsis(geq10rowsdict))
    lst.sort()
    array = np.array(lst)
    array = array.reshape(1, len(lst))
    mean = np.mean(array)
    median = np.median(array)
    q75, q25 = np.percentile(array, [75, 25])
    iqr = q75 - q25
    return mean, median, iqr

def missingRate(df: pd.DataFrame) -> float:
    """
    Returns the fraction of the DataFrame that contains NaN values

    Args:
        df: A dataframe
    
    Returns:
        A float value which is the fraction of df that is NaN values
    """
    numofNaN = df.drop(['Filename'], axis=1).isna().sum().sum()
    total_entries = df.drop(['Filename'], axis=1).count().sum() + numofNaN
    return numofNaN / total_entries