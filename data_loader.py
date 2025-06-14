import numpy as np
import tensorflow as tf
import os
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(data_dir='dataset', test_split=0.2):
    """
    Load and preprocess the dataset with stratified splitting
    
    Args:
        data_dir (str): Directory containing the dataset
        test_split (float): Proportion of data to use for testing
        
    Returns:
        tuple: (train_data, test_data, train_labels, test_labels)
    """
    try:
        # Load and preprocess data
        data = []
        labels = []
        
        # Load autism images
        autism_dir = os.path.join(data_dir, 'autism')
        autism_files = os.listdir(autism_dir)
        logger.info(f"Found {len(autism_files)} autism images")
        
        for img_name in autism_files:
            img_path = os.path.join(autism_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(128, 128)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data.append(img_array)
            labels.append(1)  # 1 for autism
            
        # Load non-autism images
        non_autism_dir = os.path.join(data_dir, 'non_autism')
        non_autism_files = os.listdir(non_autism_dir)
        logger.info(f"Found {len(non_autism_files)} non-autism images")
        
        for img_name in non_autism_files:
            img_path = os.path.join(non_autism_dir, img_name)
            img = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(128, 128)
            )
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            data.append(img_array)
            labels.append(0)  # 0 for non-autism
            
        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)
        
        # Log class distribution before splitting
        autism_count = np.sum(labels == 1)
        non_autism_count = np.sum(labels == 0)
        logger.info(f"Total dataset - Autism: {autism_count}, Non-autism: {non_autism_count}")
        
        # Normalize pixel values
        data = data.astype('float32') / 255.0
        
        # Convert labels to one-hot encoding
        labels = tf.keras.utils.to_categorical(labels, num_classes=2)
        
        # Split into train and test sets using stratified split
        train_data, test_data, train_labels, test_labels = train_test_split(
            data, labels, 
            test_size=test_split,
            stratify=labels[:, 1],  # Use autism class (1) for stratification
            random_state=42
        )
        
        # Log class distribution after splitting
        train_autism = np.sum(train_labels[:, 1] == 1)
        train_non_autism = np.sum(train_labels[:, 0] == 1)
        test_autism = np.sum(test_labels[:, 1] == 1)
        test_non_autism = np.sum(test_labels[:, 0] == 1)
        
        logger.info(f"Train set - Autism: {train_autism}, Non-autism: {train_non_autism}")
        logger.info(f"Test set - Autism: {test_autism}, Non-autism: {test_non_autism}")
        logger.info(f"Data loaded successfully. Train set: {len(train_data)}, Test set: {len(test_data)}")
        
        return train_data, test_data, train_labels, test_labels
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

if __name__ == "__main__":
    # Test the data loader
    train_data, test_data, train_labels, test_labels = load_data()
    print(f"Train data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test labels shape: {test_labels.shape}") 