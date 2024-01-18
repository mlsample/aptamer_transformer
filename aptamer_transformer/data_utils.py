import os
import pandas as pd
from collections import Counter
import numpy as np
import re
import yaml
import pickle
import warnings

# Filter out UserWarnings raised from sklearn's preprocessing module
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.preprocessing._data')


from sklearn.preprocessing import quantile_transform , StandardScaler
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from aptamer_transformer.dataset import *
from aptamer_transformer.factories_model_loss import model_config


def read_cfg(args_config):
    """
    Reads the YAML configuration file and returns a dictionary of configurations.

    Parameters:
    args_config (str): Filepath to the configuration file.

    Returns:
    dict: Dictionary containing configurations with keys as configuration names and values as configuration values.
    """
    with open(args_config, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
            working_dir = cfg['working_dir']
            model_type = cfg['model_type']
            cfg = {k: v.replace('{WORKING_DIR}', f'{working_dir}') if isinstance(v, str) else v for k, v in cfg.items()}
            cfg = {k: v.replace('{MODEL_TYPE}', f'{model_type}') if isinstance(v, str) else v for k, v in cfg.items()}
            
        except yaml.YAMLError as exc:
            print(exc)
    
    cfg =  model_config(cfg)
    
    return cfg

def read_data_files(cfg):
    """
    Reads all CSV files in the specified directory and returns a dictionary of pandas DataFrames.

    Parameters:
    cfg (dict): Configuration dictionary with 'data_directory' and 'debug' keys.

    Returns:
    dict: A dictionary where each key is the filename and the value is a DataFrame containing sequences.
    """
    directory = cfg['data_directory']
    # Get a list of all the CSV files in the directory
    data_files = sorted(os.listdir(directory), key=lambda x: int(re.findall(r'\d+', x)[0]))
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


def normalized_counters(dfs):
    """
    Normalizes the count of sequences in each DataFrame and returns a dictionary of Counters.

    Parameters:
    dfs (dict): Dictionary of pandas DataFrames, each containing sequences.

    Returns:
    dict: Dictionary of Counters, each representing normalized counts of sequences.
    """
    counter_set = {key: Counter(dfs[key]['sequence']) for key in dfs.keys()}
    for counter in counter_set.values():
        total = sum(counter.values(), 0.0)
        for key in counter:
            counter[key] /= total
    return counter_set

def get_enrichment(round_1_count, round_2_count):
    """
    Calculates the enrichment score for sequences present in both count dictionaries.

    Parameters:
    round_1_count (Counter): Counter object with sequence counts from a round.
    round_2_count (Counter): Counter object with sequence counts from another round.

    Returns:
    dict: Dictionary with sequences as keys and their enrichment scores as values.
    """
    enrichment = {}
    for seq_key in round_1_count.keys():
        if seq_key in round_2_count:
            enrichment[seq_key] = round_2_count[seq_key] / round_1_count[seq_key]
    return enrichment

def all_enrichments(counter_set):
    """
    Computes enrichment scores between all pairs of Counter objects in the input dictionary.

    Parameters:
    counter_set (dict): Dictionary of Counter objects representing sequence counts.

    Returns:
    dict: Dictionary with tuples of Counter object names as keys and enrichment score dictionaries as values.
    """
    enrichment_scores = {}
    processed_pairs = set()
    for k, v in counter_set.items():
        for key, value in counter_set.items():
            if key != k and (key, k) not in processed_pairs:
                enrichment_scores[k, key] = get_enrichment(v, value)
                processed_pairs.add((k, key))
                processed_pairs.add((key, k))
    enrichment_scores = {k: v for k, v in enrichment_scores.items() if len(v) > 0}

    return enrichment_scores

def quantile_normed_enrichment(enrichment_scores, cfg):
    """
    Applies quantile normalization to enrichment scores.

    Parameters:
    enrichment_scores (dict): Dictionary of enrichment scores.
    cfg (dict): Configuration dictionary containing 'n_quantiles' key.

    Returns:
    dict: Dictionary with quantile-normalized enrichment scores.
    """
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
    """
    Calculates the logarithmic mean of 'R' numbers extracted from the input key.

    Parameters:
    key (str): Key string containing 'R' numbers.

    Returns:
    float: Logarithmic mean of the 'R' numbers.
    """
    # Extracting 'R' numbers and converting them to integers
    r_numbers = [int(part.split('_R')[-1].split('.')[0]) for part in key]
    # Calculating the mean and then the logarithm of the mean
    return np.log(np.mean(r_numbers))

def do_round_weighting(quantile_normed_enrichment_scores):
    """
    Applies round weighting to quantile-normalized enrichment scores.

    Parameters:
    quantile_normed_enrichment_scores (dict): Dictionary with quantile-normalized enrichment scores.

    Returns:
    dict: Dictionary with weighted enrichment scores.
    """
    for key, counter_obj in quantile_normed_enrichment_scores.items():
        weight = calculate_weight(key)
        for seq in counter_obj:
            counter_obj[seq] *= weight
    return quantile_normed_enrichment_scores

def enrichment_normalization_two(df, cfg):
    """
    Applies a second round of normalization to the combined enrichment data.

    Parameters:
    df (DataFrame): DataFrame containing enrichment data.
    cfg (dict): Configuration dictionary with 'norm_2' key.

    Returns:
    DataFrame: DataFrame with normalized enrichment data.
    """
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
    """
    Loads and preprocesses enrichment data based on the provided configuration.

    Parameters:
    cfg (dict): Configuration dictionary.

    Returns:
    DataFrame: DataFrame with preprocessed enrichment data.
    """
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
    """
    Loads structure data from a specified file.

    Parameters:
    cfg (dict): Configuration dictionary with 'working_dir' key.

    Returns:
    DataFrame: DataFrame containing structure data.
    """
    with open(f'{cfg["working_dir"]}/data/nupack_strucutre_data/mfe.pickle', 'rb') as f:
        mfe = pickle.load(f)
    
    dot_bracket_struc = [mfe[key][0].structure.dotparensplus() for key in mfe.keys()]
    adjacency_matrix = [mfe[key][0].structure.matrix() for key in mfe.keys()]
    energy = [mfe[key][0].energy for key in mfe.keys()]
    
    struc_df = pd.DataFrame({'dot_bracket_struc': dot_bracket_struc, 'adjacency_matrix': adjacency_matrix, 'energy': energy})
    
    return struc_df

def load_seq_and_struc_data(cfg):
    """
    Loads and merges sequence and structure data.

    Parameters:
    cfg (dict): Configuration dictionary.

    Returns:
    DataFrame: DataFrame containing merged sequence and structure data.
    """
    seq_enrich_df = load_and_preprocess_enrichment_data(cfg)
    struc_df = load_strucutre_data(cfg)
    
    if cfg['debug'] is True:
        struc_df = struc_df.iloc[:len(seq_enrich_df)]
    
    df = pd.concat([seq_enrich_df.reset_index(drop=True), struc_df.reset_index(drop=True)], axis=1)

    return df


def load_dataset(cfg):
    """
    Loads the dataset based on the provided configuration,
    either from preprocessed data or by generating a new dataset.
    
    Parameters:
    cfg (dict): Configuration dictionary with keys for dataset loading and preprocessing options.

    Returns:
    object: A PyTorch dataset object.
    """
    if cfg['load_saved_data_set'] is not True:
        if cfg['load_saved_df'] is not False:
            df = pd.read_pickle(cfg['load_saved_df'])
        else:
            df = load_seq_and_struc_data(cfg)
        
        dna_dataset = get_pytorch_dataset(df, cfg)
    else:            
        dna_dataset = load_saved_data_set(cfg)
    
    if cfg['save_data_set'] is True:
        save_data_set_as_pickle(dna_dataset, cfg)
        
    cfg['num_tokens'] = dna_dataset.tokenizer.vocab_size
    cfg['max_seq_len'] = dna_dataset.tokenizer.model_max_length
        
    return dna_dataset


def get_pytorch_dataset(df, cfg):
    """
    Retrieves a PyTorch dataset based on the provided DataFrame and configuration.

    Parameters:
    df (DataFrame): DataFrame containing the dataset.
    cfg (dict): Configuration dictionary with 'model_config' key.

    Returns:
    object: A PyTorch dataset object.
    """
    model_config = cfg['model_config']
    if "dataset_class" in model_config:
        dataset_class = model_config['dataset_class']
        return dataset_class(df, cfg)
    else:
        raise ValueError(f"Invalid dataset_class in model_config: {model_config}")

def load_saved_data_set(cfg):
    """
    Loads a saved dataset from a pickle file specified in the configuration.

    Parameters:
    cfg (dict): Configuration dictionary with 'model_config' key.

    Returns:
    object: A PyTorch dataset object loaded from the pickle file.
    """
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
    """
    Saves a dataset as a pickle file based on the provided configuration.

    Parameters:
    dna_dataset (object): A PyTorch dataset object to be saved.
    cfg (dict): Configuration dictionary with 'model_config' key.

    Returns:
    None: The function saves the dataset and does not return a value.
    """
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
    """
    Splits the dataset into training, validation, and test sets and creates data loaders for each.

    Parameters:
    dna_dataset (object): A PyTorch dataset object.
    cfg (dict): Configuration dictionary with keys for data loader configuration.
    args (object): Argument object with 'distributed' key.

    Returns:
    tuple: A tuple containing DataLoader objects for the training, validation, and test sets, and the train sampler.
    """
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

