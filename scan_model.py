import tensorflow as tf
import numpy as np
import h5py
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_model_architecture(model_path='autism.h5'):
    """
    Analyze the deep learning model architecture
    """
    try:
        # Load the model
        logger.info(f"Loading model from {model_path}...")
        model = tf.keras.models.load_model(model_path)
        
        # Print detailed model architecture
        logger.info("\n=== Model Architecture Analysis ===")
        logger.info(f"Model Type: {type(model).__name__}")
        
        # Analyze each layer in detail
        total_params = 0
        logger.info("\nLayer-by-Layer Analysis:")
        for i, layer in enumerate(model.layers):
            logger.info(f"\nLayer {i+1}: {layer.name}")
            logger.info(f"Type: {type(layer).__name__}")
            logger.info(f"Input Shape: {layer.input_shape}")
            logger.info(f"Output Shape: {layer.output_shape}")
            params = layer.count_params()
            total_params += params
            logger.info(f"Parameters: {params:,}")
            
            # Additional details for specific layer types
            if isinstance(layer, tf.keras.layers.Conv2D):
                logger.info(f"Filters: {layer.filters}")
                logger.info(f"Kernel Size: {layer.kernel_size}")
                logger.info(f"Activation: {layer.activation.__name__}")
            elif isinstance(layer, tf.keras.layers.LSTM):
                logger.info(f"Units: {layer.units}")
                logger.info(f"Return Sequences: {layer.return_sequences}")
            elif isinstance(layer, tf.keras.layers.Dense):
                logger.info(f"Units: {layer.units}")
                logger.info(f"Activation: {layer.activation.__name__}")
        
        logger.info(f"\nTotal Model Parameters: {total_params:,}")
        
        # Get model configuration
        config = model.get_config()
        logger.info("\nModel Configuration:")
        logger.info(f"Loss Function: {model.loss}")
        logger.info(f"Optimizer: {model.optimizer.get_config()}")
        logger.info(f"Metrics: {model.metrics_names}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error analyzing model: {str(e)}")
        return None

if __name__ == "__main__":
    model = analyze_model_architecture()
    if model:
        logger.info("\nModel analysis completed successfully!")
    else:
        logger.error("Model analysis failed!") 