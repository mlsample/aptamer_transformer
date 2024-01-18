import unittest
import yaml
import os
from aptamer_transformer.data_utils import *
import pandas as pd
from unittest.mock import patch, mock_open, Mock
import random
import string


class TestDataUtils(unittest.TestCase):

    def test_read_cfg(self):
        # Path to the config.yaml file
        config_path = "aptamer_transformer/config.yaml"

        # Test if the function correctly reads and processes the config file
        cfg = read_cfg(config_path)
        self.assertIsNotNone(cfg, "The configuration should not be None")
        self.assertIn('working_dir', cfg, "Config should contain 'working_dir'")
        self.assertIn('model_type', cfg, "Config should contain 'model_type'")
        # Add more assertions as needed based on the expected contents of your config file
        
    @patch('os.listdir')
    @patch('pandas.read_csv')
    def test_read_data_files(self, mock_read_csv, mock_listdir):
        # Mocking the listdir and read_csv functions
        mock_listdir.return_value = ['data_1.csv', 'data_2.csv']
        mock_data = pd.DataFrame({'sequence': ['ATCG', 'CGTA']})
        mock_read_csv.return_value = mock_data

        # Mock configuration
        cfg = {
            'data_directory': 'some/directory',
            'debug': False
        }

        # Call the function
        dfs = read_data_files(cfg)

        # Assertions
        self.assertIsInstance(dfs, dict, "The function should return a dictionary")
        self.assertEqual(len(dfs), 2, "There should be two dataframes in the dictionary")
        self.assertTrue(all(isinstance(df, pd.DataFrame) for df in dfs.values()), "Each item in the dictionary should be a DataFrame")
        
    def test_normalized_counters(self):
        # Creating a mock DataFrame
        mock_data = pd.DataFrame({'sequence': ['ATCG', 'CGTA', 'ATCG', 'ATCG']})
        mock_dfs = {'mock_df': mock_data}

        # Calling the function
        result = normalized_counters(mock_dfs)

        # Verifying the result
        expected_result = {'mock_df': Counter({'ATCG': 0.75, 'CGTA': 0.25})}
        self.assertEqual(result, expected_result)
        
    def test_get_enrichment(self):
        round_1_count = Counter({'ATCG': 2, 'CGTA': 1})
        round_2_count = Counter({'ATCG': 4, 'CGTA': 2})

        result = get_enrichment(round_1_count, round_2_count)

        expected_result = {'ATCG': 2, 'CGTA': 2}
        self.assertEqual(result, expected_result)
        
    def test_all_enrichments(self):
        counter_set = {
            'round1': Counter({'ATCG': 2, 'CGTA': 1}),
            'round2': Counter({'ATCG': 4, 'CGTA': 2})
        }

        result = all_enrichments(counter_set)

        expected_result = {('round1', 'round2'): {'ATCG': 2, 'CGTA': 2}}
        self.assertEqual(result, expected_result)
        
    def test_quantile_normed_enrichment(self):
        enrichment_scores = {'mock_pair': {'ATCG': 2, 'CGTA': 2}}
        cfg = {'n_quantiles': 100}

        result = quantile_normed_enrichment(enrichment_scores, cfg)

        # Checking if result is a dictionary with expected keys
        self.assertIsInstance(result, dict)
        self.assertIn('mock_pair', result)

    def test_calculate_weight(self):
        key = ['HanS_R1.txt', 'HanS_R2.txt']
        result = calculate_weight(key)
        expected_result = np.log(np.mean([1, 2]))
        self.assertEqual(result, expected_result)

    # def test_do_round_weighting(self):
    #     quantile_normed_scores = {('HanS_R1.txt', 'HanS_R2.txt'): {'ATCG': 2, 'CGTA': 2}}
    #     result = do_round_weighting(quantile_normed_scores)
    #     # Assuming calculate_weight returns a constant value for the mock key
    #     mock_weight = calculate_weight(('HanS_R1.txt', 'HanS_R2.txt'))
    #     expected_result = {'mock_pair': {'ATCG': 2 * mock_weight, 'CGTA': 2 * mock_weight}}
    #     self.assertEqual(result, expected_result)

    def test_enrichment_normalization_two(self):
        df = pd.DataFrame({'Normalized_Frequency': [1, 2, 3]})
        cfg = {'norm_2': 'quantile_transform'}
        result = enrichment_normalization_two(df, cfg)
        # Test that the DataFrame is modified as expected
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Normalized_Frequency', result.columns)

    def test_load_and_preprocess_enrichment_data(self):
        cfg = {'norm_2':'quantile_transform','round_weighting': True, 'n_quantiles':2, 'num_classes': 2, 'data_directory': 'test_data'}
        mock_dfs = {
            'HanS_R1.txt': generate_mock_sequence_data(100),
            'HanS_R2.txt': generate_mock_sequence_data(100),
            'HanS_R3.txt': generate_mock_sequence_data(100),
            'HanS_R4.txt': generate_mock_sequence_data(100)
        }
        # Patching read_data_files to return mock_dfs
        with patch('aptamer_transformer.data_utils.read_data_files', return_value=mock_dfs):
            result = load_and_preprocess_enrichment_data(cfg)

        # Assertions to check if the result is as expected
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('Discretized_Frequency', result.columns)
        self.assertIn('Normalized_Frequency', result.columns)
        
    @patch('builtins.open')
    @patch('pickle.load')
    def test_load_structure_data(self, mock_pickle_load, mock_open):
        # Mock data setup
        # Mock data setup
        mock_structure = Mock(dotparensplus=lambda: 'structure1', matrix=lambda: 'matrix1')
        mock_energy = 'energy1'
        mock_item = Mock(structure=mock_structure, energy=mock_energy)
        mock_data = {'key1': [mock_item]}        
        mock_pickle_load.return_value = mock_data

        cfg = {'working_dir': 'test_dir'}
        result = load_strucutre_data(cfg)

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('dot_bracket_struc', result.columns)
        self.assertIn('adjacency_matrix', result.columns)
        self.assertIn('energy', result.columns)
        

    @patch('aptamer_transformer.data_utils.load_and_preprocess_enrichment_data')
    @patch('aptamer_transformer.data_utils.load_strucutre_data')
    def test_load_seq_and_struc_data(self, mock_load_struc, mock_load_enrich):
        # Mock data setup
        mock_load_enrich.return_value = pd.DataFrame({'seq_data': ['data1', 'data2']})
        mock_load_struc.return_value = pd.DataFrame({'struc_data': ['data3', 'data4']})

        cfg = {}
        result = load_seq_and_struc_data(cfg)
        
        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('seq_data', result.columns)
        self.assertIn('struc_data', result.columns)
        
    @patch('aptamer_transformer.data_utils.load_seq_and_struc_data')
    @patch('aptamer_transformer.data_utils.get_pytorch_dataset')
    @patch('aptamer_transformer.data_utils.load_saved_data_set')
    @patch('builtins.open')
    @patch('pickle.dump')
    def test_load_dataset(self, mock_pickle_dump, mock_open, mock_load_saved, mock_get_pytorch, mock_load_seq_struc):
        # Mock data setup
        cfg = {'load_saved_data_set': False, 'save_data_set': False}
        mock_pytorch_dataset = Mock()
        mock_get_pytorch.return_value = mock_pytorch_dataset

        result = load_dataset(cfg)

        # Assertions
        self.assertEqual(result, mock_pytorch_dataset)
        
    def test_get_pytorch_dataset(self):
        cfg = {'model_config': {'dataset_class': Mock(return_value='mock_dataset')}}
        df = pd.DataFrame({'data': [1, 2, 3]})
        result = get_pytorch_dataset(df, cfg)

        # Assertions
        self.assertEqual(result, 'mock_dataset')

    @patch('builtins.open')
    @patch('pickle.load')
    def test_load_saved_data_set(self, mock_pickle_load, mock_open):
        # Mock data setup
        cfg = {'model_config': {'dataset_class': Mock(file_path_to_pickled_dataset=lambda cfg: 'mock_file_path')}}
        mock_pickle_load.return_value = 'mock_dataset'

        result = load_saved_data_set(cfg)

        # Assertions
        self.assertEqual(result, 'mock_dataset')
        
    @patch('builtins.open')
    @patch('pickle.dump')
    def test_save_data_set_as_pickle(self, mock_pickle_dump, mock_open):
        cfg = {'model_config': {'dataset_class': Mock(file_path_to_pickled_dataset=lambda cfg: 'mock_file_path')}}
        dna_dataset = 'mock_dataset'

        result = save_data_set_as_pickle(dna_dataset, cfg)

        # Assertions
        mock_pickle_dump.assert_called_once()
        self.assertIsNone(result)

    @patch('torch.utils.data.random_split')
    @patch('torch.utils.data.DataLoader')
    def test_get_data_loaders(self, mock_dataloader, mock_random_split):
        # Mock data setup
        cfg = {'batch_size': 10, 'world_size': 1, 'rank': 0, 'seed_value': 1234, 'num_workers': 2}
        args = Mock(distributed=False)
        mock_dataset = ['data1', 'data2', 'data3']

        train_loader, val_loader, test_loader, train_sampler = get_data_loaders(mock_dataset, cfg, args)

        # Assertions
        self.assertIsNotNone(train_loader)
        self.assertIsNotNone(val_loader)
        self.assertIsNotNone(test_loader)



def generate_mock_sequence_data( num_sequences, sequence_length=4):
    nucleotides = ['A', 'T', 'C', 'G']
    sequences = [''.join(random.choices(nucleotides, k=sequence_length)) for _ in range(num_sequences)]
    return pd.DataFrame({'sequence': sequences})

if __name__ == '__main__':
    unittest.main()
