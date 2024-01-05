from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence

class DNASequenceDataSet(Dataset):
    def __init__(self, df):
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
        self.len_x = [len(tokenlized_ten) for tokenlized_ten in self.tokenized_tensor]
        self.padded_tensor = self.pad(self.tokenized_tensor)
        
        self.x = self.padded_tensor
        
        self.y = torch.Tensor(self.labels)
        # self.y = self.y.type(torch.LongTensor)
        self.y = self.y.float()

    
    def __getitem__(self, idx):
        
        batch_len_x = self.len_x  # Length of the sequence
        
        return self.x[idx], self.y[idx], batch_len_x[idx]
    
    def __len__(self):
        return len(self.data)
    
    def tokenize(self, data):
        # Nucleotide to integer mapping, including 'N'
        nucleotide_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'P':5}
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
