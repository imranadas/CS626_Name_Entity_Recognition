# model_loader.py

import os
import pickle
from logger_config import setup_logger

logger = setup_logger('model_loader')

def load_model_components(model_dir):
    """
    Load trained model components from the specified directory
    
    Args:
        model_dir (str): Directory containing model artifacts
        
    Returns:
        tuple: (model, vectorizer, scaler) loaded from files
        
    Raises:
        FileNotFoundError: If model files are not found
        Exception: For other errors during loading
    """
    logger.info(f"Loading model components from directory: {model_dir}")
    
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    
    required_files = [
        (model_path, 'model'),
        (vectorizer_path, 'vectorizer'),
        (scaler_path, 'scaler')
    ]
    
    try:
        # Check if all required files exist
        missing_files = [name for path, name in required_files if not os.path.exists(path)]
        if missing_files:
            raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
        
        # Load components
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            logger.info("Successfully loaded model")
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            logger.info("Successfully loaded vectorizer")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            logger.info("Successfully loaded scaler")
        
        return model, vectorizer, scaler
    
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        raise
