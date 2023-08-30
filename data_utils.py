import os
import pandas as pd
from collections import Counter
import numpy as np

def read_data_files(directory):
    """
    Reads all the CSV files in the specified directory and returns a list of dataframes.
    Each dataframe contains a single column named 'sequence' with the sequences from the corresponding CSV file.
    """
    # Get a list of all the CSV files in the directory
    data_files = os.listdir(directory)
    data_path = [os.path.join(directory, file) for file in data_files]
    
    dfs = {}
    # Read each CSV file into a dataframe and append it to the list
    for idx, file in enumerate(data_path):
        df = pd.read_csv(file, header=None, names=['sequence'])
        dfs[data_files[idx]] = df

    return dfs

def get_min_max_normalized_frequency(combined_counter):
    # Extract the raw frequencies
    raw_frequencies = np.array(list(combined_counter.values()))
    
    # Calculate Min-Max normalization constants
    min_val = np.min(raw_frequencies)
    max_val = np.max(raw_frequencies)
    
    # Perform Min-Max normalization
    normalized_frequencies = {key: (value - min_val) / (max_val - min_val) for key, value in combined_counter.items()}
    
    # Sort by normalized frequency
    sorted_items = sorted(normalized_frequencies.items(), key=lambda item: item[1], reverse=True)
    
    return sorted_items

def load_and_preprocess_data(directory):
    dfs = read_data_files(directory)
    counter_set = [Counter(dfs[key]['sequence']) for key in dfs.keys()]
    counters_combined = sum(counter_set, Counter())
    sorted_normalized_items = get_min_max_normalized_frequency(counters_combined)
    df = pd.DataFrame(sorted_normalized_items, columns=['Sequence', 'Normalized_Frequency'])
    return df
