import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import logging
from data_loader import load_data
import json
from train_model import ResidualBlock  # Import the custom layer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path):
    """
    Load the trained model with custom objects
    """
    try:
        # Register the custom ResidualBlock layer
        custom_objects = {
            'ResidualBlock': ResidualBlock
        }
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logger.info(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def evaluate_model(model, test_data, test_labels):
    """Evaluate model performance"""
    try:
        # Log initial shape
        logger.info(f"Initial test data shape: {test_data.shape}")
        
        # Add temporal dimension if not present
        if len(test_data.shape) == 4:  # (batch, height, width, channels)
            test_data = np.expand_dims(test_data, axis=1)  # Add temporal dimension
            logger.info(f"Shape after adding temporal dimension: {test_data.shape}")
        
        # Reshape data to match model's expected input shape (None, 15, 128, 128, 3)
        if test_data.shape[1] == 1:
            # If we have single frames, repeat them to create a sequence of 15 frames
            test_data = np.repeat(test_data, 15, axis=1)
        elif test_data.shape[1] < 15:
            # If we have fewer than 15 frames, pad with the last frame
            padding = np.repeat(test_data[:, -1:], 15 - test_data.shape[1], axis=1)
            test_data = np.concatenate([test_data, padding], axis=1)
        elif test_data.shape[1] > 15:
            # If we have more than 15 frames, take the first 15
            test_data = test_data[:, :15]
        
        logger.info(f"Final test data shape: {test_data.shape}")
        
        # Verify shape matches model's expected input
        expected_shape = (None, 15, 128, 128, 3)
        if test_data.shape[1:] != expected_shape[1:]:
            raise ValueError(f"Data shape {test_data.shape} does not match expected shape {expected_shape}")
        
        # Get predictions
        predictions = model.predict(test_data)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1)
        
        # Calculate metrics
        accuracy = np.mean(predicted_classes == true_classes)
        logger.info(f"Model Accuracy: {accuracy:.4f}")
        
        # Log additional metrics
        cm = confusion_matrix(true_classes, predicted_classes)
        precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
        recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        return predictions, predicted_classes, true_classes
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('analysis_plots/confusion_matrix.png')
        plt.close()
        logger.info("Confusion matrix plot saved")
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {str(e)}")

def plot_roc_curve(y_true, y_pred_proba):
    """Plot ROC curve"""
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.savefig('analysis_plots/roc_curve.png')
        plt.close()
        logger.info("ROC curve plot saved")
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {str(e)}")

def plot_precision_recall_curve(y_true, y_pred_proba):
    """Plot precision-recall curve"""
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig('analysis_plots/precision_recall_curve.png')
        plt.close()
        logger.info("Precision-recall curve plot saved")
    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {str(e)}")

def analyze_model():
    """
    Analyze the trained model's performance and architecture
    """
    try:
        # Create analysis_plots directory if it doesn't exist
        os.makedirs('analysis_plots', exist_ok=True)
        
        # Load the model with custom objects
        model = load_model('autism_model.h5')
        
        # Load test data
        _, test_data, _, test_labels = load_data()
        
        # Log input shapes for debugging
        logger.info(f"Test data shape before reshaping: {test_data.shape}")
        
        # Evaluate model
        predictions, predicted_classes, true_classes = evaluate_model(model, test_data, test_labels)
        
        # Generate plots
        plot_confusion_matrix(true_classes, predicted_classes)
        plot_roc_curve(true_classes, predictions)
        plot_precision_recall_curve(true_classes, predictions)
        
        logger.info("Model analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model analysis: {str(e)}")
        raise

if __name__ == "__main__":
    analyze_model()