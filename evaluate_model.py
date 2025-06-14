import tensorflow as tf
import numpy as np
import os
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from train_model import load_and_preprocess_data, process_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(model_path='autism.h5', data_dir='Dataset', batch_size=2):
    """
    Evaluate the model's accuracy and performance metrics
    """
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Load and preprocess test data
        logger.info("Loading test data...")
        file_paths, labels = load_and_preprocess_data(data_dir)
        
        # Convert labels to one-hot encoding
        labels = tf.keras.utils.to_categorical(labels, num_classes=2)
        
        # Process data in batches
        all_predictions = []
        all_true_labels = []
        
        logger.info("Evaluating model...")
        for i in range(0, len(file_paths), batch_size):
            batch_files = file_paths[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            # Process batch
            X_batch, y_batch = process_batch(batch_files, batch_labels)
            
            # Get predictions
            batch_predictions = model.predict(X_batch, verbose=0)
            all_predictions.extend(batch_predictions)
            all_true_labels.extend(y_batch)
            
            # Clear memory
            del X_batch, y_batch, batch_predictions
            tf.keras.backend.clear_session()
        
        # Convert predictions to class labels
        y_pred = np.argmax(all_predictions, axis=1)
        y_true = np.argmax(all_true_labels, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, target_names=['Autism', 'Non-Autism'])
        
        # Print results
        logger.info("\n=== Model Evaluation Results ===")
        logger.info(f"Overall Accuracy: {accuracy:.4f}")
        logger.info("\nConfusion Matrix:")
        logger.info(conf_matrix)
        logger.info("\nClassification Report:")
        logger.info(class_report)
        
        # Plot confusion matrix using matplotlib
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks([0, 1], ['Autism', 'Non-Autism'])
        plt.yticks([0, 1], ['Autism', 'Non-Autism'])
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Calculate class-wise accuracy
        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        logger.info("\nClass-wise Accuracy:")
        logger.info(f"Autism: {class_accuracies[0]:.4f}")
        logger.info(f"Non-Autism: {class_accuracies[1]:.4f}")
        
        return accuracy, conf_matrix, class_report
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        return None, None, None

if __name__ == "__main__":
    accuracy, conf_matrix, class_report = evaluate_model()
    if accuracy is not None:
        logger.info("\nModel evaluation completed successfully!")
    else:
        logger.error("Model evaluation failed!") 