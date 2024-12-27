import numpy as np
import pandas as pd

def prepare_data(training_data, new_data):
    res = new_data.copy()
    res['SpecialProperty'] = res['blood_type'].isin(['O+','B+'])
    res.drop(columns=['blood_type'],inplace = True)
    num_cols = res.select_dtypes(include=['int', 'float']).columns
    res[num_cols] = res[num_cols].fillna(res[num_cols].median())
    features = res.filter(like='PCR').columns.tolist()
    for feature in features:
        mean = training_data[feature].mean()
        std = training_data[feature].std()
        max = training_data[feature].max()
        min = training_data[feature].min()
        normalised_mm_colomn = f'normalized_minmax_{feature}'
        normalised_sd_colomn = f'normalized_standard_{feature}'
        res[normalised_mm_colomn] = [(x-min)/(max-min) for x in res[feature]]
        res[normalised_sd_colomn] = [(x-mean)/std for x in res[feature]]
    return res
