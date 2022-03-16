from tqdm import tqdm

import time
import sys
import pandas as pd
import numpy as np

# (batch_size, days, lat_bin, lon_bin, crime_t)
DAYS = 30
LAT_GRID_SIZE = 16
LON_GRID_SIZE = 16
CRIME_TYPE = 1

BIN_LAT_COL = 'binned_latitude'
BIN_LON_COL = 'binned_longitude'

def generate_batches(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n]
        
def coord_to_grid(dataset, lat_col, lon_col):
    dataset[BIN_LAT_COL] = pd.cut(dataset[lat_col], LAT_GRID_SIZE, labels=False, retbins=False).fillna(0).astype(int)
    dataset[BIN_LON_COL] = pd.cut(dataset[lon_col], LON_GRID_SIZE, labels=False, retbins=False).fillna(0).astype(int)
    return dataset

def feature_reduce(dataset, features, sort_by, drop_duplicates_by='Date'):
    dataset = dataset[features].copy()
    dataset.sort_values(by=[sort_by], ascending=False, inplace=True)
    dataset.drop_duplicates(subset=[drop_duplicates_by], inplace=True)
    return dataset
    
def convert_to_image_data(dataset, crime_col, lat_col=BIN_LAT_COL, lon_col=BIN_LON_COL):
    start = time.time()
    inputs = []
    outputs = []
    batches = list(generate_batches([data for _, data in dataset.iterrows()], DAYS))

    for batch_number, batch in tqdm(enumerate(batches), desc='Converting...'):
        current = time.time()
        if False: # print progress, TODO: use tqdm progress bar
            print(f"""
            \rBatches Generated: {batch_number + 1}
            \rPercentage Generated: {((batch_number + 1)*100)//len(batches)}%
            \rBatches Remaining: {len(batches) - (batch_number + 1)} {''}""", end="")
            sys.stdout.flush()

        features = np.zeros((DAYS, LAT_GRID_SIZE, LON_GRID_SIZE, CRIME_TYPE))
        for day, row in enumerate(batch): # Only one day in batch
            features[day, row[lat_col], row[lon_col]] = 1
            target = row[crime_col]
            inputs.append(features)
            outputs.append(target)

    inputs = np.concatenate([np.expand_dims(s, axis=0) for s in inputs], axis=0)
    outputs = np.concatenate([np.expand_dims(s, axis=0) for s in outputs], axis=0)
    
    print(f"Time Elapsed: {int((current - start)//60)} min")
    return { 
        'inputs': inputs,
        'outputs': outputs,
        'batches': len(batches)
    }
    
        
