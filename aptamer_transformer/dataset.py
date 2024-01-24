from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import os
import numpy as np
from sklearn.preprocessing import OneHotEncoder
        
# class SeqDataset()

        
class SeqClassifierDataset(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()        
        tokenized_data = self.tokenizer(space_sep_seqs, padding=True, return_tensors="pt")
        
        self.tokenized_seqs = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        self.y  = df.Discretized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_seqs[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seqs)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_classifier_dataset.pickle'

    
class SeqRegressionDataset(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()        
        tokenized_data = self.tokenizer(space_sep_seqs, padding=True, return_tensors="pt")
        
        self.tokenized_seqs = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        self.y  = df.Normalized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_seqs[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seqs)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_regression_dataset.pickle'


class SeqBertDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()
        
        self.tokenized_seqs = space_sep_seqs
        self.y  = np.zeros((len(space_sep_seqs))).tolist()
        
    def __getitem__(self, idx):
        
        return self.tokenized_seqs[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seqs)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_bert_dataset.pickle'

    
class StructClassifierDataset(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['dot_bracket_tokenizer_path'])
        
        space_sep_struct = df.dot_bracket.apply(lambda x: ' '.join(x)).to_list()
        tokenized_data = self.tokenizer(space_sep_struct, padding=True, return_tensors="pt")
        
        self.tokenized_struc = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        
        self.y  = df.Discretized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_struc[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_struc)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/struct_classifier_dataset.pickle'

    
class StructRegressionDataset(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['dot_bracket_tokenizer_path'])
        
        space_sep_struct = df.dot_bracket.apply(lambda x: ' '.join(x)).to_list()
        tokenized_data = self.tokenizer(space_sep_struct, padding=True, return_tensors="pt")
        
        self.tokenized_struc = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        
        self.y  = df.Normalized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_struc[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_struc)

    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/struct_regression_dataset.pickle'
    

class SeqStructEnerMatrixRegressionDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()
        space_sep_struct = df.dot_bracket.apply(lambda x: ' '.join(x)).to_list()
        seq_structs_white_space = [(seq , struct) for seq, struct in zip(space_sep_seqs, space_sep_struct)]        
        
        tokenized_data = self.tokenizer(seq_structs_white_space, padding=True, return_tensors="pt")

        self.tokenized_seq_struc = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        self.energy = torch.Tensor(df.energy)
        
        
        self.y  = df.Normalized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_seq_struc[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seq_struc)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_struct_regression_dataset.pickle'

class SeqStructRegressionDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()
        space_sep_struct = df.dot_bracket.apply(lambda x: ' '.join(x)).to_list()
        seq_structs_white_space = [(seq , struct) for seq, struct in zip(space_sep_seqs, space_sep_struct)]        
        
        tokenized_data = self.tokenizer(seq_structs_white_space, padding=True, return_tensors="pt")

        self.tokenized_seq_struc = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        
        self.y  = df.Normalized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_seq_struc[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seq_struc)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_struct_regression_dataset.pickle'

    
class SeqStructClassifierDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()
        space_sep_struct = df.dot_bracket.apply(lambda x: ' '.join(x)).to_list()
        seq_structs_white_space = [(seq , struct) for seq, struct in zip(space_sep_seqs, space_sep_struct)]        
        
        tokenized_data = self.tokenizer(seq_structs_white_space, padding=True, return_tensors="pt")

        self.tokenized_seq_struc = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        
        self.y  = df.Discretized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_seq_struc[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seq_struc)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_struct_classification_dataset.pickle'

    
class SeqStructBertDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(cfg['seq_struct_tokenizer_path'])

        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()
        space_sep_struct = df.dot_bracket.apply(lambda x: ' '.join(x)).to_list()
        seq_structs_white_space = [(seq, struct) for seq, struct in zip(space_sep_seqs, space_sep_struct)]        
                
        self.tokenized_seqs = self.tokenizer(seq_structs_white_space)
        self.y = np.zeros((len(space_sep_seqs))).tolist()
        
    def __getitem__(self, idx):
        
        return self.tokenized_seqs[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seqs)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_struct_bert_dataset.pickle'


class XGBoostDataset(Dataset):
    def __init__(self, df_filtered, cfg):

        max_seq_length = max(df_filtered['Sequence'].apply(len))
        max_dot_bracket_length = max(df_filtered['dot_bracket'].apply(len))

        sequence_tokenized = np.array([self.tokenize_sequence(seq, max_seq_length) for seq in df_filtered['Sequence']])
        dot_bracket_tokenized = np.array([self.tokenize_sequence(seq, max_dot_bracket_length, '.') for seq in df_filtered['dot_bracket']])

        # One-hot encoding the tokenized sequences
        onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        sequence_encoded = onehot_encoder.fit_transform(sequence_tokenized)
        dot_bracket_encoded = onehot_encoder.fit_transform(dot_bracket_tokenized)

        # Flatten the encoded arrays
        sequence_flattened = sequence_encoded.reshape(df_filtered.shape[0], -1)
        dot_bracket_flattened = dot_bracket_encoded.reshape(df_filtered.shape[0], -1)

        # Feature 2 - energy
        energy = df_filtered['energy'].values.reshape(-1, 1)

        # Feature 4 - structure_matrix
        max_matrix_size = max(df_filtered['strucutre_matrix'].apply(lambda x: len(x)))
        padded_structure_matrix = df_filtered['strucutre_matrix'].apply(
            lambda matrix: np.pad(matrix, (0, max_matrix_size - len(matrix)), mode='constant')
        )
        structure_matrix = np.array(padded_structure_matrix.tolist())
        flattened_structure_matrix = np.array([matrix.flatten() for matrix in structure_matrix])

        # Concatenate all features
        X = np.hstack((sequence_flattened, dot_bracket_flattened, energy, flattened_structure_matrix))
        
        self.X = torch.Tensor(X).unsqueeze(-1)
        
        self.y = torch.Tensor(df_filtered['Normalized_Frequency'].values)
        
        self.tokenizer = PsudoObject()
        self.tokenizer.vocab_size = self.X.shape[1]
        self.tokenizer.model_max_length = self.X.shape[1]
        
    def __getitem__(self, idx):
        
        return self.X[idx], self.y[idx]
    
    def __len__(self):
        return len(self.X)

    def tokenize_sequence(self, seq, max_length, padding_char='N'):
        seq += padding_char * (max_length - len(seq))  # Pad the sequence
        return list(seq)

    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/xgboost_dataset.pickle'


class PsudoObject:
    pass