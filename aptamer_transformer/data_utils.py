import os
import pandas as pd
from collections import Counter
import numpy as np
import re
import yaml
import pickle

from sklearn.preprocessing import quantile_transform , StandardScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from aptamer_transformer.dataset import *


def read_cfg(args_config):
    # Read the YAML configuration file
    with open(args_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
            working_dir = cfg['working_dir']
            model_type = cfg['model_type']
            cfg = {k: v.replace('{WORKING_DIR}', f'{working_dir}') if isinstance(v, str) else v for k, v in cfg.items()}
            cfg = {k: v.replace('{MODEL_TYPE}', f'{model_type}') if isinstance(v, str) else v for k, v in cfg.items()}

        except yaml.YAMLError as exc:
            print(exc)
    
    return cfg

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
            df = pd.read_csv(file, header=None, names=['sequence'], nrows=10000)
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

def enrichment_normalization_two(df, cfg):
    # Using 'assign' to modify 'Normalized_Frequency' and sorting values
    if cfg['norm_2'] == 'quantile_transform':
        df = df.assign(
            Normalized_Frequency=lambda x: quantile_transform(x['Normalized_Frequency'].values.reshape(-1, 1)).reshape(-1)
        )
    elif cfg['norm_2'] == 'standard_scaler':
        scaler = StandardScaler()
        df = df.assign(
            Normalized_Frequency=lambda x: scaler.fit_transform(x['Normalized_Frequency'].values.reshape(-1, 1)).reshape(-1)
        )
    return df


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
    
    df = enrichment_normalization_two(df, cfg)
    
    df['Discretized_Frequency'] = pd.qcut(df['Normalized_Frequency'], q=cfg['num_classes'], labels=False)

    return df

def load_strucutre_data(cfg):
    with open(f'{cfg["working_dir"]}/data/nupack_strucutre_data/mfe.pickle', 'rb') as f:
        mfe = pickle.load(f)
    
    dot_bracket_struc = [mfe[key][0].structure.dotparensplus() for key in mfe.keys()]
    adjacency_matrix = [mfe[key][0].structure.matrix() for key in mfe.keys()]
    energy = [mfe[key][0].energy for key in mfe.keys()]
    
    struc_df = pd.DataFrame({'dot_bracket_struc': dot_bracket_struc, 'adjacency_matrix': adjacency_matrix, 'energy': energy})
    
    return struc_df

def load_seq_and_struc_data(cfg):
    
    seq_enrich_df = load_and_preprocess_enrichment_data(cfg)
    struc_df = load_strucutre_data(cfg)
    
    df = pd.concat([seq_enrich_df.reset_index(drop=True), struc_df.reset_index(drop=True)], axis=1)

    return df


def load_dataset(cfg):
    
    if cfg['load_saved_data_set'] is not True:
        df = load_seq_and_struc_data(cfg)
        dna_dataset = get_pytorch_dataset(df, cfg)
    else:            
        dna_dataset = load_saved_data_set(cfg)
    
    if cfg['save_data_set'] is True:
        save_data_set_as_pickle(dna_dataset, cfg)
        
    return dna_dataset


def get_pytorch_dataset(df, cfg):
    model_config = cfg['model_config']
    if "dataset_class" in model_config:
        dataset_class = model_config['dataset_class']
        return dataset_class(df, cfg)
    else:
        raise ValueError(f"Invalid dataset_class in model_config: {model_config}")

def load_saved_data_set(cfg):
    model_config = cfg['model_config']
    if "dataset_class" in model_config:
        dataset_class = model_config['dataset_class']
        filepath = dataset_class.file_path_to_pickled_dataset(cfg)
        
        with open(filepath, 'rb') as f:
            dna_dataset = pickle.load(f)
        
        return dna_dataset
    else:
        raise ValueError(f"Invalid dataset_class in model_config: {model_config}")
        

def save_data_set_as_pickle(dna_dataset, cfg):
    model_config = cfg['model_config']
    if "dataset_class" in model_config:
        dataset_class = model_config['dataset_class']
        filepath = dataset_class.file_path_to_pickled_dataset(cfg)
        
        with open(filepath, 'wb') as f:
            pickle.dump(dna_dataset, f, pickle.HIGHEST_PROTOCOL)
        
        return None
    else:
        raise ValueError(f"Invalid dataset_class in model_config: {model_config}")


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

