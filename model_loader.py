# model_loader.py

import os
import pickle
from logger_config import setup_logger

logger = setup_logger('model_loader')

def load_model(model_dir):
    """
    Load a trained model and vectorizer from the specified directory
    
    Args:
        model_dir (str): Directory containing model artifacts
        
    Returns:
        tuple: (model, vectorizer) loaded from files
        
    Raises:
        FileNotFoundError: If model files are not found
        Exception: For other errors during loading
    """
    logger.info(f"Loading model from directory: {model_dir}")
    
    model_path = os.path.join(model_dir, 'model.pkl')
    vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
    
    try:
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError("Model files not found")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            logger.info("Successfully loaded model")
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
            logger.info("Successfully loaded vectorizer")
        
        return model, vectorizer
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
