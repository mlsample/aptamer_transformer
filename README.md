# Aptamer Transformer Project

The Aptamer Transformer Project is a machine learning application designed to generate DNA sequences using a transformer model. The project is implemented in Python and utilizes PyTorch for model training and inference. The transformer model is trained on a dataset of DNA sequences and is capable of generating new sequences based on the learned patterns.

## Project Structure

The project is organized into several Python scripts and Jupyter notebooks:

- `config.yaml`: This file houses the configuration parameters for the model, training, and data.
- `data_utils.py`: This script contains utility functions for loading and preprocessing the data.
- `dataset.py`: This script defines the `DNASequenceDataSet` class, which is used for loading the data into PyTorch.
- `main.py`: This is the primary script for training the model.
- `model.py`: This script defines the `DNATransformerEncoder` and `DNAXTransformerEncoder` classes, which are the main model classes.
- `training_utils.py`: This script contains utility functions for training, validating, and testing the model.
- `run_inference.ipynb`: This Jupyter notebook is used for running inference on the trained model.
- `metric_analysis.ipynb`: This Jupyter notebook is used for analyzing the performance of the model.

## Usage

To train the model, you can use the `torchrun` command. For single node distributed training, use the following command:

`torchrun --standalone --nnodes=1 --nproc-per-node=$N_GPUS main.py --config config.yaml --distributed`

To run on a single GPU, use the following command:

`torchrun main.py --config config.yaml`

The `main.py` script is the main entry point for training the model. It initializes the model and the optimizer, and runs the training loop. If the load_last_checkpoint option in the configuration is set to True, it loads the last saved model checkpoint before starting the training.

The script also handles distributed training. If the distributed argument is passed, the model is wrapped with DistributedDataParallel, which parallelizes the model across multiple GPUs.

The `model.py` script defines the `DNATransformerEncoder` and `DNAXTransformerEncoder` classes, which are the main model classes. The model takes a DNA sequence as input and outputs a regression or classification result, depending on the model_task in the configuration.

The `metric_analysis.ipynb` notebook can be used to analyze the performance of the model. It loads the model's predictions and the ground truth labels, and computes various metrics.