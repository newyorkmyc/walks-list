'''
Scripts and functions to read in CSVs, clean data, and make visualizations.

Main functions:
---------------
read_all_csvs: create Pandas DataFrames for all files in a directory
'''

import geopandas as gpd
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

MONTHS = {1:'Jan', 2:'Feb', 3:'March', 4:'April', 5:'May', 6:'June',
          7:'July', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}

PARKS_MAPPING = {'B': 'Brooklyn',
                 'X': 'Bronx',
                 'M': 'Manhattan',
                 'R': 'Staten Island',
                 'Q': 'Queens'}


def clean_data(df):
    '''Chain together all cleaning/standardizing functions into
    one overall cleaning function. This will make it easier to 
    add steps along the way.
    
    TO DO: Might want to make this into an sklearn
    pre-processing pipeline.'''
    df = clean_nulls(df)
    df = clean_cols(df)
    df = get_dates(df)
    
    # Fix a couple of misspelled park names
    if 'ParkName' in df.columns:
        df['ParkName'] = (df['ParkName']
                          .str.replace("Wolfe\\'s Pond Park", "Wolfe's Pond Park", regex=False)
                          .str.replace('Randalls Island', "Randall's Island Park", regex=False))

    # Remove rows with unknown species ("sp.", "spp.", NaN, etc.)
    if 'Species' in df.columns:
        df = remove_unknown_sp(df)
        
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


def get_observations_data(compound_dates=False):
    '''Return a df with data by individual observations,
    merging several csvs together.
    '''
    dfs = read_all_csvs('./data', verbose=False)
    observations = (dfs['observations']
                    .merge(dfs['mushroom'][['MushroomID','BroadGroupID','Genus','Species']], 
                           on='MushroomID', how='left')
                    .merge(dfs['broadgroups'][['BroadGroupID','BroadGroupName']], on='BroadGroupID', how='left')
                    .merge(dfs['walks'][['WalkID','ParkID','WalkDate']])
                    .drop(['Notes','LinkToINat','NewToPark','NewToCity','ParkID',
                           'WalkID','ObservationID','DateCreated','DateModified'], axis=1)
                    .dropna(subset=['Species']))
    observations['Date'] = observations['WalkDate'].dt.normalize()
    observations['Month'] = observations['WalkDate'].dt.month.map(MONTHS)
    observations['Week'] = observations.apply(lambda x: x['WalkDate'].isocalendar()[1], axis=1)
    observations['Year'] = observations.apply(lambda x: x['WalkDate'].isocalendar()[0], axis=1)
    if compound_dates:
        observations['Month_Year'] = observations.apply(lambda x: f"{x['Month']}, {x['Year']}", axis=1)
        observations['Week_Year'] = observations.apply(lambda x: f"Week {x['Week']}, {x['Year']}", axis=1)
        observations['Quarter'] = observations['WalkDate'].dt.quarter
        observations['Quarter_Year'] = observations.apply(lambda x: f"Q{x['Quarter']}, {x['Year']}", axis=1)
    return remove_duplicate_obs(remove_unknown_sp(observations))
    

def get_parks_data():
    '''Return a df with data by parks, merging several
    csvs together.
    '''
    dfs = read_all_csvs('./data', verbose=False)
    parks_data = (
        dfs['observations']
        .merge(dfs['parks'], on='ParkID', how='left')
        .merge(dfs['mushroom'][['MushroomID','BroadGroupID','Genus','Species']], on='MushroomID', how='left')
        .merge(dfs['broadgroups'][['BroadGroupID','BroadGroupName']], on='BroadGroupID', how='left')
        .drop(['LinkToINat','ParkID','ObservationID','DateCreated','DateModified'], axis=1)
    )
    parks_data['Genus'] = parks_data['Genus'].str.strip()
    parks_data['Species'] = parks_data['Species'].str.strip()
    parks_data['FullName'] = parks_data.apply(lambda x: f"{x['Genus']} {x['Species']}", axis=1)
    return remove_duplicate_obs(remove_unknown_sp(parks_data))


def groupby_multipoly(df, by, aggfunc="first"):
    '''Take directly from
    https://stackoverflow.com/questions/64811011/geopandas-converting-single-polygons-to-multipolygon-keeping-individual-polygo
    '''
    data = df.drop(labels=df.geometry.name, axis=1)
    aggregated_data = data.groupby(by=by).agg(aggfunc)

    # Process spatial component
    def merge_geometries(block):
        return MultiPolygon(block.values)

    g = df.groupby(by=by, group_keys=False)[df.geometry.name].agg(
        merge_geometries
    )

    # Aggregate
    aggregated_geometry = gpd.GeoDataFrame(g, geometry=df.geometry.name, crs=df.crs)
    # Recombine
    aggregated = aggregated_geometry.join(aggregated_data)
    return aggregated
    

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
        if file[-4:]!='.csv':
            # raise TypeError(f'File {file} is not a csv.')
            continue
        if verbose:
            print(f'[*] Reading in {file}...')
        dfs[f"{file.split('.')[0].lower().replace(' ','_')}"] = clean_data(pd.read_csv(dir_path+file))
    return dfs


def remove_duplicate_obs(df, keep_var=True):
    '''Many walks list the same mushroom species multiple times. 
    Often this is related to species being unknown, but this may 
    also be due to errors in data entry. 
    
    Additional context from an email from Ethan Crenson on 11/29/24:
        "We usually list species once even if found multiple times 
        over the course of a walk. The exceptions might be:
        
        1- varieties. The lists are structured to deal with genus 
        and species. When we find Amanita brunnescens and 
        Amanita brunnescens var. pallida they both go on the list. 
        But the var. pallida part goes in the notes. 
        
        2- undetermined species left at genus can appear twice. 
        Usually there are also differences noted in the notes field. 
        I.e. Gymnopus sp. (hairy stem) Gymnopus sp. (smooth stem).
        
        3-the occasional mistake. This should not be extensive in 
        the lists."

    As such, we will remove duplicate species listings from the 
    data for this analysis except where indicated in the
    observation notes.
    '''
    if keep_var:
        # Split df by repeated rows
        dupe_rows = df.duplicated(subset=['WalkID', 'MushroomID'], keep=False)
        single_obs_rows = df.loc[~dupe_rows]
        repeated_obs_rows = df.loc[dupe_rows]
        
        # From https://stackoverflow.com/questions/60928060/pandas-drop-duplicates-where-condition
        repeated_obs_rows = repeated_obs_rows[(~(repeated_obs_rows[['WalkID', 'MushroomID']].duplicated()) 
                                               | repeated_obs_rows['Notes'].str.contains('var.'))]

        # Join all rows back together
        return pd.concat([single_obs_rows,repeated_obs_rows]).sort_index()
    else:
        return df.drop_duplicates(subset=['WalkID','MushroomID'], keep='first')


def remove_unknown_sp(df):
    '''Remove all rows with unknown species, identified in
    the data as 'sp.' or 'spp.'
    '''
    df = df.dropna(subset=['Species'])
    df = df.loc[(~df['Species'].str.contains('sp.', na=False)) & (~df['Species'].str.contains('spp.', na=False))]
    df = df.loc[df['Species']!='sp']
    return df