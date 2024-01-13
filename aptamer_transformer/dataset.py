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
    

class AptamerBertDataSet(Dataset):
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
        self.y  = df.Normalized_Frequency
        
    def __getitem__(self, idx):
        
        return self.tokenized_seqs[idx], self.y[idx]
    
    def __len__(self):
        return len(self.tokenized_seqs)
    
    
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
    

class DNAStructEncoderDataSet(Dataset):
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
    
    
    
    
##############
# Depreciated
##############
    
class DNASequenceDataSet(Dataset):
    def __init__(self, df, cfg):
        """
        Initialize the DNASequenceDataSet.
        
        Parameters:
            data (Tensor): The tokenized and embedded DNA sequences.
            labels (Tensor, optional): The labels corresponding to each sequence.
            vocab (dict, optional): The vocabulary mapping each nucleotide to a unique integer.
        """
        self.data = df.Sequence
        self.labels = df.Normalized_Frequency
        
        self.tokenized_tensor = self.tokenize(self.data)
        
        max_seq_len = max([len(tokenlized_ten) for tokenlized_ten in self.tokenized_tensor])
        self.len_x = [len(tokenlized_ten) for tokenlized_ten in self.tokenized_tensor]
        
        self.padded_tensor = self.pad(self.tokenized_tensor)
        self.pad_mask = ~self.create_mask(len(self.data), max_seq_len, self.len_x)

        self.x = torch.Tensor(self.padded_tensor)
        self.y = torch.Tensor(self.labels)

        
        cfg['num_tokens'] = self.vocab_size
        cfg['max_seq_len'] = max_seq_len

    
    def __getitem__(self, idx):
                
        return self.x[idx], self.pad_mask[idx], self.y[idx]
    
    def __len__(self):
        return len(self.data)
    
    def tokenize(self, data):
        # Nucleotide to integer mapping, including 'N'
        nucleotide_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'P':5}
        self.vocab_size = len(nucleotide_to_int)
        # nucleotide_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'P':5, 'CLS':6}
        # Tokenize sequences
        tokenized_sequences = data.apply(lambda x: [nucleotide_to_int.get(n, 4) for n in x])  # Default to 4 ('N') if nucleotide not in dictionary
        # tokenized_sequences = data.apply(lambda x: [nucleotide_to_int['CLS']] + [nucleotide_to_int.get(n, 4) for n in x])  # Default to 4 ('N') if nucleotide not in dictionary
        # Convert to PyTorch tensor
        tokenized_tensor = [torch.tensor(seq, dtype=torch.long) for seq in tokenized_sequences]

        return tokenized_tensor
    
    def pad(self, tokenized_tensor):
        # Pad sequences
        padded_tensor = pad_sequence(tokenized_tensor, batch_first=True, padding_value=5)  # Padding value set to 5 ('P')

        return padded_tensor
    
    def create_mask(self, len_data, max_seq_len, len_x):
        mask_pad = torch.zeros(len_data, max_seq_len).bool()
        for (idx, len_seq) in enumerate(len_x): 
            mask_pad[idx, len_seq:] = 1
        return mask_pad