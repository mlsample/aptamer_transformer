from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import os
import numpy as np

        
class SeqClassifierDataset(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()        
        tokenized_data = tokenizer(space_sep_seqs, padding=True, return_tensors="pt")
        
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
        tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()        
        tokenized_data = tokenizer(space_sep_seqs, padding=True, return_tensors="pt")
        
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
        
        tokenizer = AutoTokenizer.from_pretrained(cfg['dot_bracket_tokenizer_path'])
        
        space_sep_struct = df.dot_bracket_struc.apply(lambda x: ' '.join(x)).to_list()
        tokenized_data = tokenizer(space_sep_struct, padding=True, return_tensors="pt")
        
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
        
        tokenizer = AutoTokenizer.from_pretrained(cfg['dot_bracket_tokenizer_path'])
        
        space_sep_struct = df.dot_bracket_struc.apply(lambda x: ' '.join(x)).to_list()
        tokenized_data = tokenizer(space_sep_struct, padding=True, return_tensors="pt")
        
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
    

class SeqStructDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the AptamerBert.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        
        tokenizer = AutoTokenizer.from_pretrained(cfg['dot_bracket_tokenizer_path'])
        
        space_sep_seqs = df.Sequence.apply(lambda x: ' '.join(x)).to_list()
        space_sep_struct = df.dot_bracket_struc.apply(lambda x: ' '.join(x)).to_list()
        
        # tokenized_seq_data = tokenizer(space_sep_seqs, padding=True, return_tensors="pt")
        tokenized_data = tokenizer(space_sep_struct, padding=True, return_tensors="pt")

        
        self.tokenized_struc = tokenized_data['input_ids']
        self.attn_masks = tokenized_data['attention_mask']
        
        self.y  = df.Normalized_Frequency
        
    def __getitem__(self, idx):

        return self.tokenized_struc[idx], self.attn_masks[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_struc)
    
    @classmethod
    def file_path_to_pickled_dataset(cls, cfg):
        return f'{cfg["working_dir"]}/data/saved_processed_data/pickled/seq_struct_dataset.pickle'

    
