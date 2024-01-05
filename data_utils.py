import os
import pandas as pd
from collections import Counter
import numpy as np
import re
from sklearn.preprocessing import quantile_transform 
from dataset import DNASequenceDataSet
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

def read_data_files(cfg):
    """
    Reads all the CSV files in the specified directory and returns a list of dataframes.
    Each dataframe contains a single column named 'sequence' with the sequences from the corresponding CSV file.
    """
    directory = cfg['data_directory']
    # Get a list of all the CSV files in the directory
    data_files = sorted(os.listdir(directory), key=lambda x: int(re.findall(r'\d+', x)[0]))
    data_path = [os.path.join(directory, file) for file in data_files]
    
    dfs = {}
    # Read each CSV file into a dataframe and append it to the list
    if cfg['debug'] is True:
        for idx, file in enumerate(data_path):
            df = pd.read_csv(file, header=None, names=['sequence'], nrows=1000000)
            dfs[data_files[idx]] = df
    else:
        for idx, file in enumerate(data_path):
            df = pd.read_csv(file, header=None, names=['sequence'])
            dfs[data_files[idx]] = df
        
    return dfs


def normalized_counters(dfs):
    counter_set = {key: Counter(dfs[key]['sequence']) for key in dfs.keys()}
    for counter in counter_set.values():
        total = sum(counter.values(), 0.0)
        for key in counter:
            counter[key] /= total
    return counter_set

def get_enrichment(round_1_count, round_2_count):
    enrichment = {}
    for seq_key in round_1_count.keys():
        if seq_key in round_2_count:
            enrichment[seq_key] = round_2_count[seq_key] / round_1_count[seq_key]
    return enrichment

def all_enrichments(counter_set):
    enrichment_scores = {}
    processed_pairs = set()
    for k, v in counter_set.items():
        for key, value in counter_set.items():
            if key != k and (key, k) not in processed_pairs:
                enrichment_scores[k, key] = get_enrichment(v, value)
                processed_pairs.add((k, key))
                processed_pairs.add((key, k))
    return enrichment_scores

def quantile_normed_enrichment(enrichment_scores, cfg):
    quantile_normed_enrichment_scores = {}
    for k, v in enrichment_scores.items():
        quantile_normed_enrichment_scores[k] = {}
        v_array = np.array(list(v.values())).reshape(-1, 1)
        seq_array = np.array(list(v.keys()))
        scores = quantile_transform(v_array, copy=True, n_quantiles=cfg['n_quantiles']).reshape(-1)
        
        for seq, score in  zip(v.keys(), scores):
            quantile_normed_enrichment_scores[k][seq] = score
            
    return quantile_normed_enrichment_scores

def calculate_weight(key):
    # Extracting 'R' numbers and converting them to integers
    r_numbers = [int(part.split('_R')[-1].split('.')[0]) for part in key]
    # Calculating the mean and then the logarithm of the mean
    return np.log(np.mean(r_numbers))

def do_round_weighting(quantile_normed_enrichment_scores):
    for key, counter_obj in quantile_normed_enrichment_scores.items():
        weight = calculate_weight(key)
        for seq in counter_obj:
            counter_obj[seq] *= weight
    return quantile_normed_enrichment_scores

def load_and_preprocess_enrichment_data(cfg):
    dfs = read_data_files(cfg)

    counter_set = normalized_counters(dfs)
    enrichment_scores = all_enrichments(counter_set)
    quantile_normed_enrichment_scores = quantile_normed_enrichment(enrichment_scores, cfg)
    
    # Applying weights if round_weighting is True
    if cfg['round_weighting'] is True:
        quantile_normed_enrichment_scores = do_round_weighting(quantile_normed_enrichment_scores)
                
    combined = sum((Counter(d) for d in quantile_normed_enrichment_scores.values()), Counter())
    df = pd.DataFrame.from_dict(combined, orient='index', columns=['Normalized_Frequency']).reset_index().rename(columns={'index': 'Sequence'})
    
    if cfg['model_task'] == 'classification':
        df['Normalized_Frequency'] = df['Normalized_Frequency'].apply(lambda x: 1 if x >= cfg['classification_threshold'] else 0)

    # Using 'assign' to modify 'Normalized_Frequency' and sorting values
    df = df.assign(
        Normalized_Frequency=lambda x: quantile_transform(x['Normalized_Frequency'].values.reshape(-1, 1)).reshape(-1)
    )

    return df

def load_dataset(cfg):
    if cfg['load_saved_data_set'] is not False:
        if cfg['model_task'] == 'classification':
            df = pd.read_hdf('dna_dataset_classification.h5', 'df')
        elif cfg['model_task'] == 'regression':
            df = pd.read_hdf('dna_dataset_regression.h5', 'df')
        
        dna_dataset = DNASequenceDataSet(df)

    else:            
        df = load_and_preprocess_enrichment_data(cfg)
        dna_dataset = DNASequenceDataSet(df)
    
    if cfg['save_data_set'] is True:
        if cfg['model_task'] == 'classification':
            df.to_hdf('dna_dataset_classification.h5', key='df', mode='w')
        elif cfg['model_task'] == 'regression':
            df.to_hdf('dna_dataset_regression.h5', key='df', mode='w')
        
    return dna_dataset

def get_data_loaders(dna_dataset, cfg, args):
    train_size = int(0.7 * len(dna_dataset))
    val_size = int(0.15 * len(dna_dataset))
    test_size = len(dna_dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dna_dataset, [train_size, val_size, test_size])

    if args.distributed:
        train_sampler = DistributedSampler(train_set, num_replicas=cfg['world_size'], rank=cfg['rank'], seed=cfg['seed_value'])
        val_sampler = DistributedSampler(val_set, num_replicas=cfg['world_size'], rank=cfg['rank'], seed=cfg['seed_value'])
        test_sampler = DistributedSampler(test_set, num_replicas=cfg['world_size'], rank=cfg['rank'], seed=cfg['seed_value'])
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    train_loader = DataLoader(train_set, batch_size=cfg['batch_size'], sampler=train_sampler, num_workers=cfg['num_workers'])
    val_loader = DataLoader(val_set, batch_size=cfg['batch_size'], sampler=val_sampler, num_workers=cfg['num_workers'])
    test_loader = DataLoader(test_set, batch_size=cfg['batch_size'], sampler=test_sampler, num_workers=cfg['num_workers'])
    
    return train_loader, val_loader, test_loader, train_sampler


##########################################################
#Deprecated functions
##########################################################

def count_one_nucleotide_away(dfs):
    """
    Count the number of sequences that are one nucleotide away from each other.
    
    Parameters:
        dfs (dict): Dictionary of DataFrames with 'sequence' column.
        
    Returns:
        int: Number of sequences that are one nucleotide away.
    """
    # Create a list of Counters for each DataFrame
    counter_set = [Counter(dfs[key]['sequence']) for key in dfs.keys()]
    
    # Initialize count of sequences that are one nucleotide away
    count = 0
    
    # Iterate through each Counter object
    for counter in counter_set:
        # Iterate through each sequence in the Counter object
        for seq1 in counter.keys():
            # Compare with every other sequence in the Counter object
            for seq2 in counter.keys():
                if seq1 != seq2:
                    # Calculate the Hamming distance between seq1 and seq2
                    hamming_distance = sum(el1 != el2 for el1, el2 in zip(seq1, seq2))
                    
                    # Check if the sequences are one nucleotide away
                    if hamming_distance == 1:
                        count += 1
    
    return count

def fix_mislabed_nucleotides(cfg):
    dfs = read_data_files(cfg)
    result = count_one_nucleotide_away(dfs)
    return result


def read_data_files_no_r1(cfg):
    """
    Reads all the CSV files in the specified directory and returns a list of dataframes.
    Each dataframe contains a single column named 'sequence' with the sequences from the corresponding CSV file.
    """
    directory = cfg['data_directory']
    # Get a list of all the CSV files in the directory
    data_files = os.listdir(directory)
    data_files = [file for file in data_files if 'HanS_R1.txt' not in file]
    print(data_files)
    data_path = [os.path.join(directory, file) for file in data_files]
    
    
    dfs = {}
    # Read each CSV file into a dataframe and append it to the list
    if cfg['debug'] is True:
        for idx, file in enumerate(data_path):
            df = pd.read_csv(file, header=None, names=['sequence'], nrows=100000)
            dfs[data_files[idx]] = df
    else:
        for idx, file in enumerate(data_path):
            df = pd.read_csv(file, header=None, names=['sequence'])
            dfs[data_files[idx]] = df
        
    return dfs


def load_and_preprocess_weighted_frequency_data_no_r1(cfg):
    dfs = read_data_files_no_r1(cfg)
    
    files = list(dfs.keys())
    round_number = [int(re.findall(r'\d+', name)[0]) for name in files]
    
    counter_set = [Counter(dfs[key]['sequence']) for key in dfs.keys()]
    
    weight_factor = [np.log(rounds) for rounds in round_number]
    
    weighted_counter_set = []
    for counts, factor in zip(counter_set, weight_factor):
        weighted_counts = Counter({k: v * factor for k, v in counts.items()})
        weighted_counter_set.append(weighted_counts)     
    
    counters_combined = sum(weighted_counter_set, Counter())
    
    counters_list = [(k, v) for k, v in counters_combined.items()]
    
    # sorted_normalized_items = get_min_max_normalized_frequency(counters_combined)
    df = pd.DataFrame(counters_list, columns=['Sequence', 'Normalized_Frequency'])
    
    minimun = df['Normalized_Frequency'].min()
    maximum = df['Normalized_Frequency'].max()
    
    df['Normalized_Frequency'] = df['Normalized_Frequency'].apply(lambda x: normalize_between_a_and_b(x, -1, 1, minimun, maximum))
    
    df = df.sort_values(by='Normalized_Frequency', ascending=False).reset_index()
    return df



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

def load_and_preprocess_data(cfg):
    dfs = read_data_files(cfg)
        
    counter_set = [Counter(dfs[key]['sequence']) for key in dfs.keys()]
    
    counters_combined = sum(counter_set, Counter())
    sorted_normalized_items = get_min_max_normalized_frequency(counters_combined)
    df = pd.DataFrame(sorted_normalized_items, columns=['Sequence', 'Normalized_Frequency'])
    return df


def load_and_preprocess_weighted_frequency_data(cfg):
    dfs = read_data_files(cfg)

    files = list(dfs.keys())
    round_number = [int(re.findall(r'\d+', name)[0]) for name in files]
    
    counter_set = [Counter(dfs[key]['sequence']) for key in dfs.keys()]

    weight_factor = [np.log(rounds) for rounds in round_number]
    
    weighted_counter_set = []
    for counts, factor in zip(counter_set, weight_factor):
        log_normalized = Counter({k: np.log(1 + v)  for k, v in counts.items()})
        
        mean_log = np.mean(list(log_normalized.values()))
        std_log = np.std(list(log_normalized.values()))
        z_score_normalized = Counter({k: ((v - mean_log)/std_log) for k, v in counts.items()})
        
        weighted_counts = Counter({k: v * factor for k, v in z_score_normalized.items()})
        weighted_counter_set.append(weighted_counts)     
    
    counters_combined = sum(weighted_counter_set, Counter())
    
    counters_list = [(k, v) for k, v in counters_combined.items()]
    
    # sorted_normalized_items = get_min_max_normalized_frequency(counters_combined)
    df = pd.DataFrame(counters_list, columns=['Sequence', 'Normalized_Frequency'])
    
    minimun = df['Normalized_Frequency'].min()
    maximum = df['Normalized_Frequency'].max()
    
    # df['Normalized_Frequency'] = df['Normalized_Frequency'].apply(lambda x: normalize_between_a_and_b(x, -1, 1, minimun, maximum))
    
    df = df.sort_values(by='Normalized_Frequency', ascending=False).reset_index()
    return df



def normalize_between_a_and_b(x, a, b, minimun, maximum):
    return ((b - a) * ((x - minimun) / (maximum - minimun))) + a

def undo_normalization(x, a, b, minimun, maximum):
    return ((x - a) * ((maximum - minimun) / (b - a))) + minimun