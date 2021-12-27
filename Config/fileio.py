import recordRetrieval
from typing import Any, List
import pickle
import os


def listOfPositiveRecordsDir() -> None:
    """Creates a pkl file to save all positive records

    Args:
        None

    Returns:
        None
    """
    if not os.path.isfile('listOfPositiveRecords.pkl'):
        lst = recordRetrieval.findPositiveFiles()
        lst.sort()
        with open("listOfPositiveRecords.pkl", "wb") as f:
            pickle.dump(lst, f)


def listOfAllRecordsDir() -> None:
    """Creates a pkl file to save all records
    
    Args:
        None

    Returns:
        None
    """
    if not os.path.isfile('listOfAllRecords.pkl'):
        lst = recordRetrieval.retrieveAllFiles()
        with open("listOfAllRecords.pkl", "wb") as f:
            pickle.dump(lst, f)


def StraightDumpDir(x: Any, name: str) -> None:
    """Used to create and save pkl files that can be saved right away without 
    any further processing.

    Args:
        x: An object to be saved
        name: Name of pkl file to be created and saved
    
    Returns:
        None
    """
    with open(name, 'wb') as f:
        pickle.dump(x, f)


def pklOpener(x: str) -> Any:
    """Opens the given pkl file

    Args:
        x: Name of pkl file to be opened
    
    Returns:
        A python object stored in that file
    """
    with open(x, "rb") as f:
        return pickle.load(f)
