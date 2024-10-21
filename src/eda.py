'''
Scripts and functions to read in CSVs, clean data, and make visualizations.

Main functions:
---------------
read_all_csvs: create Pandas DataFrames for all files in a directory
'''

import matplotlib as mpl
import numpy as np
import os
import pandas as pd

from matplotlib.colors import ListedColormap

# Set up color palatte to match report
# https://www.canva.com/design/DAGHBYpCA1o/IZLR-wTXmEJ1EfS4vLGjmA/edit?utm_content=DAGHBYpCA1o&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton
COLORS_DICT = {
    'gray':'#413934',
    'gold':'#be9530',
    'pink':'#e1839a',
    'orange':'#cf5530',
    'tan':'#e8e3d6'
}
COLORS = ['#413934', '#be9530', '#e1839a', '#cf5530', '#e8e3d6']
nymyc_cmap = ListedColormap(COLORS, name="nymyc_cmap")
nymyc_cmap_r = nymyc_cmap.reversed()


def clean_data(df):
    '''Chain together all cleaning/standardizing functions into
    one overall cleaning function. This will make it easier to 
    add steps along the way.
    
    TO DO: Might want to make this into an sklearn
    pre-processing pipeline.'''
    df = clean_nulls(df)
    df = clean_cols(df)
    df = get_dates(df)
    return df


def clean_nulls(df):
    '''Make sure all versions of missings values are converted
    to actual Pandas NaNs.
    '''
    return df.replace('na', np.nan).replace('0000-00-00 00:00:00',np.nan)


def clean_cols(df):
    '''Make sure all column names are CamelCase with the first
    letter capitalized.'''
    new_cols = [c[0].upper() + c[1:] for c in df.columns]
    df.columns = new_cols
    return df


def get_dates(df):
    '''Convert all date columns to datetime datatypes.'''
    for col in df.columns:
        if 'date' in col.lower().strip():
            df[col] = pd.to_datetime(df[col])
    return df
    

def read_all_csvs(dir_path, verbose=True):
    '''A function to read all csvs in a given folder into
    Pandas DataFrames and standardize column names.
    
    Args:
    ----------
    dir_path (string): A filepath to a folder.
    verbose (boolean): Optional; whether to print the name
        of the file being ingested.
        Default = True
    
    Returns:
    ----------
    A dictionary of dataframes where the keys are normalized
    filenames.
    '''
    if not dir_path[-1]=='/':
        dir_path+='/'
    if not dir_path[:3]=='../':
        dir_path = '../' + dir_path
    dfs = {}
    for file in os.listdir(dir_path):
        if verbose:
            print(f'[*] Reading in {file}...')
        if file[-4:]!='.csv':
            raise TypeError(f'File {file} is not a csv.')
        dfs[f"{file.split('.')[0].lower().replace(' ','_')}"] = clean_data(pd.read_csv(dir_path+file))
    return dfs
    

