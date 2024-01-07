import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pickle
from collections import Counter

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, matthews_corrcoef
from scipy.special import softmax
from sklearn.preprocessing import label_binarize

def evaluate_classification(y_true, y_pred_probs):
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    c_matrix = confusion_matrix(y_true, y_pred)

    return accuracy, precision, recall, fscore, c_matrix

def plot_classification_metrics(y_true, y_pred_probs, class_names=None):
    """
    Plots confusion matrix, and prints accuracy, precision, recall, and F1-score.

    :param y_true: Array of true class labels
    :param y_pred_probs: Array of predicted probabilities for each class
    :param class_names: List of class names for the confusion matrix
    """
    # Use evaluate_classification to calculate metrics and confusion matrix
    accuracy, precision, recall, fscore, confusion_matrix = evaluate_classification(y_true, y_pred_probs)

    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {fscore:.4f}")

    plt.show()

def plot_mean_loss(file_path, num_batches):
    # Read data from JSON file
    with open(file_path, 'r') as f:
        loss_data = json.load(f)

    # Helper function to calculate mean loss for every num_batches
    def mean_loss(loss_list, num_batches):
        return [np.mean(loss_list[i:i + num_batches]) for i in range(0, len(loss_list), num_batches)]

    # Calculate mean loss
    train_loss_means = mean_loss(loss_data['train_loss'], num_batches)
    val_loss_means = mean_loss(loss_data['val_loss'], 48)

    # Generate the plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss_means, label='Mean Training Loss')
    plt.plot(val_loss_means, label='Mean Validation Loss')
    plt.title('Mean Training and Validation Loss Over Time')
    plt.xlabel('Every {} Batches'.format(num_batches))
    plt.ylabel('Mean Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def load_test_predictions(filename='test_predictions.pkl'):
    with open(filename, 'rb') as f:
        y_true_list, y_pred_list = pickle.load(f)
    return y_true_list, y_pred_list

def evaluate_regression_model(y_true, y_pred):
    # Calculating metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    explained_variance = explained_variance_score(y_true, y_pred)

    # Printing metrics
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R-squared (Coefficient of Determination): {r2}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")
    print(f"Explained Variance Score: {explained_variance}")

    return mse, rmse, mae, r2, mape, explained_variance

def process_data(y_preds, target):
    """
    Process the predictions to align them for metric calculation, using only the masked positions.
    Args:
    - y_preds (np.array): Predicted logits from the model for each position in each sequence.
    - target (np.array): Array indicating masked positions and true token IDs for these positions.
    
    Returns:
    - np.array: Flattened array of predicted nucleotide IDs for masked positions.
    - np.array: Flattened array of true nucleotide IDs for masked positions.
    """
    # Convert logits to class predictions
    y_preds_class = np.argmax(y_preds, axis=-1)

    # Initialize lists to store processed predictions and true labels for masked positions
    processed_preds = []
    processed_true = []

    # Process each sequence
    for i in range(len(target)):
        for j, true_token in enumerate(target[i]):
            if true_token != -100:  # Check for masked positions
                processed_preds.append(y_preds_class[i, j])
                processed_true.append(true_token)

    return np.array(processed_preds), np.array(processed_true)

def plot_metrics(processed_true, processed_preds):
    """
    Calculate accuracy, precision, recall, and F1 score.
    Args:
    - processed_true (np.array): True labels for masked positions.
    - processed_preds (np.array): Predictions for masked positions.
    
    Returns:
    - dict: Dictionary containing calculated metrics.
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(processed_true, processed_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(processed_true, processed_preds, average='weighted')
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    # Names of the metrics
    metric_names = list(metrics.keys())

    # Values of the metrics
    metric_values = [metrics[name] for name in metric_names]

    # Creating the bar plot
    plt.figure(figsize=(10, 6))  # You can adjust the figure size as needed
    plt.bar(metric_names, metric_values, color='skyblue')

    # Adding titles and labels
    plt.title('NLP Model Performance Metrics')
    plt.ylabel('Score')
    plt.ylim(0, 1)  # Assuming your metric values are between 0 and 1

    # Adding value labels on top of each bar
    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.01, f'{value:.2f}', ha='center')

    # Display the plot
    plt.show()
    
    return None

def plot_confusion_matrix(processed_true, processed_preds):
    conf_mat = confusion_matrix(processed_true, processed_preds, labels=np.unique(processed_true))

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'C', 'G', 'T'], yticklabels=['A', 'C', 'G', 'T'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    return None

from sklearn.metrics import accuracy_score, confusion_matrix

def per_token_metrics(processed_true, processed_preds):
    """
    Calculate and print per-token accuracy, sensitivity, and specificity.
    Args:
    - processed_true (np.array): True labels.
    - processed_preds (np.array): Predicted labels.
    """
    nucleotide = {4: 'A', 5: 'C', 6: 'G', 7: 'T'}
    classes = np.unique(processed_true)
    # Initialize dictionary to hold metrics
    metrics = {}

    # Binarize the labels for one-vs-rest computation
    true_binarized = label_binarize(processed_true, classes=classes)
    preds_binarized = label_binarize(processed_preds, classes=classes)
    
    # Calculate confusion matrix once for all classes
    conf_mat = confusion_matrix(processed_true, processed_preds, labels=classes)
    
    # Calculate metrics for each class
    for i, class_label in enumerate(classes):
        class_metrics = {}
        
        # Accuracy
        token_true = processed_true[processed_true == class_label]
        token_preds = processed_preds[processed_true == class_label]
        class_metrics['accuracy'] = accuracy_score(token_true, token_preds)
        
        # Sensitivity (Recall) and Specificity
        tp = conf_mat[i, i]
        fn = np.sum(conf_mat[i, :]) - tp
        fp = np.sum(conf_mat[:, i]) - tp
        tn = np.sum(conf_mat) - (tp + fn + fp)
        
        class_metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) != 0 else 0
        class_metrics['specificity'] = tn / (tn + fp) if (tn + fp) != 0 else 0
        
        # Matthews Correlation Coefficient (MCC)
        class_metrics['mcc'] = matthews_corrcoef(true_binarized[:, i], preds_binarized[:, i])
        
        # Add metrics to the main dictionary
        metrics[nucleotide[class_label]] = class_metrics

    # Print metrics in a pretty format
    for nucleotide, class_metrics in metrics.items():
        print(f"\nMetrics for nucleotide {nucleotide}:")
        print(f"  Accuracy: {class_metrics['accuracy']:.4f}")
        print(f"  Sensitivity: {class_metrics['sensitivity']:.4f}")
        print(f"  Specificity: {class_metrics['specificity']:.4f}")
        print(f"  MCC: {class_metrics['mcc']:.4f}")

def extract_masked_logits(y_preds, target):
    """
    Extract logits for only the masked tokens.
    Args:
    - y_preds (np.array): Predicted logits from the model for each position in each sequence.
    - target (np.array): Array indicating masked positions and true token IDs for these positions.
    
    Returns:
    - np.array: Array of logits for the masked positions.
    """
    masked_logits = []

    # Iterate over each sequence and position
    for i in range(len(target)):
        for j, true_token in enumerate(target[i]):
            if true_token != -100:  # Check for masked positions
                masked_logits.append(y_preds[i, j])

    return np.array(masked_logits)

def plot_roc_auc_from_logits(masked_logits, processed_true):
    """
    Calculate and plot ROC curve and AUC from logits for each class using a one-vs-rest approach.
    Args:
    - masked_logits (np.array): Logits for the masked positions.
    - processed_true (np.array): True labels for the masked positions.
    - classes (np.array): Array of class labels.
    """
    classes = np.unique(processed_true)
    # Apply softmax to convert logits to probabilities
    probabilities = softmax(masked_logits, axis=1)

    # Binarize the labels for one-vs-rest computation
    true_binarized = label_binarize(processed_true, classes=classes)

    plt.figure(figsize=(10, 8))
    nucleotide = {4: 'A', 5: 'C', 6: 'G', 7: 'T'}

    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(true_binarized[:, i], probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{nucleotide[class_label]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')  # Diagonal line for reference
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Nucleotide (Using Logits)')
    plt.legend(loc='lower right')
    plt.show()
    
def find_common_error_subsequences_around_masked(y_true_tokenized, y_preds_tokenized, target, window_size=5):
    # Identify the masked positions and where predictions were incorrect
    masked_positions = np.where(target != -100)
    incorrect_predictions = np.where(y_true_tokenized[masked_positions] != y_preds_tokenized[masked_positions])

    # Collect the error subsequences around masked positions
    error_subsequences_with_pos = []
    for idx in zip(*incorrect_predictions):
        seq_idx, pos = masked_positions[0][idx], masked_positions[1][idx]
        
        # Extract the subsequence around the masked position
        start = max(0, pos - window_size // 2)
        end = min(y_true_tokenized.shape[1], start + window_size)
        subseq = y_true_tokenized[seq_idx, start:end]
        
        # Record the position of the masked token within the window
        relative_pos = pos - start
        
        error_subsequences_with_pos.append((tuple(subseq), relative_pos))

    # Count the frequency of each error subsequence with position
    subseq_counts_with_pos = Counter(error_subsequences_with_pos)
    
    # Find the most common error subsequences with positions
    common_error_subsequences_with_pos = subseq_counts_with_pos.most_common()
    
    return common_error_subsequences_with_pos

def plot_error_position_distribution(common_error_subsequences_with_pos):
    """
    Plot the distribution of error positions within the subsequences.
    
    Args:
    - common_error_subsequences_with_pos (list): List of common error subsequences with position and count.
    """
    # We're only interested in the position of the error, which is the second element in the tuple
    error_positions = [pos for (subseq, pos), count in common_error_subsequences_with_pos]

    # Count the frequency of each error position
    error_position_counts = Counter(error_positions)

    # Prepare data for plotting
    positions = list(range(max(error_position_counts.keys()) + 1))
    counts = [error_position_counts[pos] for pos in positions]

    # Plot distribution of error positions
    plt.figure(figsize=(10, 5))
    plt.bar(positions, counts, color='skyblue', edgecolor='grey')
    plt.xlabel('Position in Subsequence')
    plt.ylabel('Error Count')
    plt.title('Distribution of Error Positions within Subsequences')
    plt.xticks(positions)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


