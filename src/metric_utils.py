import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error, explained_variance_score
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, multilabel_confusion_matrix, confusion_matrix
import seaborn as sns

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
