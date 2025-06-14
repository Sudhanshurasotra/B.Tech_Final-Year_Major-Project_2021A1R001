import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, LSTM, TimeDistributed, Dropout, BatchNormalization, Input, Add, Activation, GlobalAveragePooling2D, Multiply, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import glob
import gc  # For garbage collection
import json
import tensorflow_addons as tfa
from tensorflow.keras.regularizers import l2
from data_loader import load_data  # Import the load_data function

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set numpy to use float32
np.set_printoptions(precision=3)
tf.keras.backend.set_floatx('float32')

# Limit GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logger.error(f"GPU memory configuration error: {e}")

def extract_frames(video_path, max_frames=15, target_size=(128, 128)):  # Reduced frames and size
    """
    Extract frames from a video file with memory management
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize frame
        frame = cv2.resize(frame, target_size)
        # Normalize and convert to float32
        frame = (frame / 255.0).astype(np.float32)
        frames.append(frame)
        frame_count += 1
        
    cap.release()
    
    # If we have fewer frames than max_frames, pad with the last frame
    while len(frames) < max_frames:
        frames.append(frames[-1] if frames else np.zeros((*target_size, 3), dtype=np.float32))
    
    return np.array(frames, dtype=np.float32)

def load_image(image_path, target_size=(128, 128)):  # Reduced size
    """
    Load and preprocess a single image
    """
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = (np.array(img) / 255.0).astype(np.float32)
        return img_array
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {str(e)}")
        return None

def load_and_preprocess_data(data_dir, max_frames=15, target_size=(128, 128), batch_size=2):
    """
    Load and preprocess the dataset with batch processing and memory management
    """
    # Get all file paths first
    all_files = []
    all_labels = []
    
    for class_idx, class_name in enumerate(['autism', 'non_autism']):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            logger.warning(f"Directory {class_dir} does not exist")
            continue
            
        logger.info(f"Processing {class_name} class...")
        
        # Get all files
        video_files = glob.glob(os.path.join(class_dir, '*.mp4')) + \
                     glob.glob(os.path.join(class_dir, '*.avi')) + \
                     glob.glob(os.path.join(class_dir, '*.mov'))
        image_files = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                     glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                     glob.glob(os.path.join(class_dir, '*.png'))
        
        all_files.extend(video_files + image_files)
        all_labels.extend([class_idx] * (len(video_files) + len(image_files)))
    
    if not all_files:
        raise ValueError("No valid data found in the specified directory")
    
    # Convert to numpy arrays
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    
    logger.info(f"Total number of files: {len(all_files)}")
    logger.info(f"Number of autism samples: {np.sum(all_labels == 0)}")
    logger.info(f"Number of non-autism samples: {np.sum(all_labels == 1)}")
    
    return all_files, all_labels

def process_batch(file_paths, labels, max_frames=15, target_size=(128, 128)):
    """
    Process a batch of files to manage memory
    """
    batch_X = []
    batch_y = []
    
    for file_path, label in zip(file_paths, labels):
        try:
            if file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                frames = extract_frames(file_path, max_frames, target_size)
            else:
                img_array = load_image(file_path, target_size)
                if img_array is not None:
                    frames = np.array([img_array] * max_frames, dtype=np.float32)
                else:
                    continue
                    
            batch_X.append(frames)
            batch_y.append(label)
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
            
    return np.array(batch_X, dtype=np.float32), np.array(batch_y)

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = Conv2D(filters, kernel_size, padding='same')
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.conv2 = Conv2D(filters, kernel_size, padding='same')
        self.bn2 = BatchNormalization()
        self.projection = Conv2D(filters, 1, padding='same')
        self.add = Add()
        self.act2 = Activation('relu')
    
    def call(self, inputs):
        # First conv block
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Add residual connection
        if inputs.shape[-1] == self.filters:
            residual = inputs
        else:
            residual = self.projection(inputs)
        
        x = self.add([residual, x])
        x = self.act2(x)
        return x
    
    def compute_output_shape(self, input_shape):
        # The output shape is the same as input shape except for the last dimension
        return input_shape[:-1] + (self.filters,)
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config

def create_hybrid_model(input_shape=(15, 128, 128, 3), num_classes=2):
    """
    Create an optimized hybrid CNN-LSTM model with attention for autism detection
    """
    # Input layer with temporal dimension
    inputs = Input(shape=input_shape)
    
    # Process each frame through CNN
    x = TimeDistributed(Conv2D(64, 7, strides=2, padding='same'))(inputs)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(MaxPooling2D(3, strides=2, padding='same'))(x)
    
    # Residual blocks with increasing filters
    x = TimeDistributed(ResidualBlock(64, name='res_block_1'))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    
    x = TimeDistributed(ResidualBlock(128, name='res_block_2'))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    
    x = TimeDistributed(ResidualBlock(256, name='res_block_3'))(x)
    x = TimeDistributed(MaxPooling2D(2))(x)
    x = TimeDistributed(Dropout(0.2))(x)
    
    # Attention mechanism
    attention = TimeDistributed(Conv2D(1, 1, activation='sigmoid'))(x)
    x = TimeDistributed(Multiply())([x, attention])
    
    # Global average pooling
    x = TimeDistributed(GlobalAveragePooling2D())(x)
    
    # LSTM part for temporal feature processing
    x = LSTM(256, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = LSTM(128)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers with regularization
    x = Dense(128, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

def train_model(data_dir='dataset', batch_size=8):
    """
    Train the hybrid CNN-LSTM model
    
    Args:
        data_dir (str): Directory containing the dataset
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (trained model, training history)
    """
    try:
        # Load and preprocess data using the imported function
        train_data, test_data, train_labels, test_labels = load_data(data_dir)
        
        # Create data generators
        train_generator = data_generator(train_data, train_labels, batch_size, is_training=True)
        val_generator = data_generator(test_data, test_labels, batch_size, is_training=False)
        
        # Calculate steps per epoch
        train_steps = len(train_data) // batch_size
        val_steps = len(test_data) // batch_size
        
        # Create and compile model
        logger.info("Creating model...")
        model = create_hybrid_model()
        
        # Print model summary
        model.summary(print_fn=logger.info)
        
        # Initial learning rate
        initial_learning_rate = 0.001
        
        # Learning rate schedule function
        def lr_schedule(epoch, lr):
            if epoch < 10:
                return lr
            else:
                return lr * tf.math.exp(-0.1)
        
        # Optimizer with gradient clipping
        optimizer = Adam(learning_rate=initial_learning_rate, clipnorm=1.0)
        
        # Compile model with weighted loss
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )
        
        # Callbacks
        callbacks = [
            # Early stopping
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            # Model checkpoint
            tf.keras.callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max'
            ),
            # Learning rate scheduler
            tf.keras.callbacks.LearningRateScheduler(
                lr_schedule,
                verbose=1
            ),
            # TensorBoard logging
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1
            )
        ]
        
        # Train the model
        logger.info("Starting training...")
        history = model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=50,
            validation_data=val_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Training completed successfully")
        return model, history
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

def predict_with_threshold(model, image, threshold=0.5):
    """
    Make prediction with adjusted threshold for better sensitivity
    """
    prediction = model.predict(image)
    autism_prob = prediction[0][1]  # Probability of autism class
    
    # Apply threshold
    if autism_prob >= threshold:
        return "Autism Detected", autism_prob
    else:
        return "No Autism Detected", 1 - autism_prob

if __name__ == "__main__":
    # Set your data directory here
    DATA_DIR = "dataset"  # Change this to your dataset directory
    
    # Create dataset directory if it doesn't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'autism'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 'non_autism'), exist_ok=True)
    
    logger.info(f"Please place your data in the following structure:")
    logger.info(f"{DATA_DIR}/")
    logger.info(f"├── autism/")
    logger.info(f"│   ├── video1.mp4")
    logger.info(f"│   ├── image1.jpg")
    logger.info(f"│   └── ...")
    logger.info(f"└── non_autism/")
    logger.info(f"    ├── video1.mp4")
    logger.info(f"    ├── image1.jpg")
    logger.info(f"    └── ...")
    
    # Check if data exists
    if not any(os.listdir(os.path.join(DATA_DIR, 'autism'))) or not any(os.listdir(os.path.join(DATA_DIR, 'non_autism'))):
        logger.warning("No data found in the dataset directories. Please add your data before running the training.")
    else:
        
        # Train the model with optimized batch size
        model, history = train_model(DATA_DIR, batch_size=8)
        logger.info("Training completed. Model saved as 'autism_model.h5'") 