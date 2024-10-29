# inference.py

from model_loader import load_model
from logger_config import setup_logger
from utils import prepare_data, preprocess_sentence

logger = setup_logger('inference')

class NERPredictor:
    def __init__(self, model_dir):
        """
        Initialize the NER predictor by loading the trained model and vectorizer
        
        Args:
            model_dir (str): Directory containing model artifacts
            
        Raises:
            FileNotFoundError: If model files are not found
            Exception: For other initialization errors
        """
        logger.info(f"Initializing NER predictor with model directory: {model_dir}")
        
        try:
            self.model, self.vectorizer = load_model(model_dir)
            logger.info("Successfully initialized NER predictor")
        except Exception as e:
            logger.error(f"Failed to initialize NER predictor: {str(e)}")
            raise
    
    def predict(self, sentence):
        """
        Predict named entities in a sentence
        
        Args:
            sentence (str): Input sentence
            
        Returns:
            tuple: (tagged_text, list of (token, prediction) pairs)
        """
        logger.info(f"Processing sentence: {sentence}")
        try:
            original_tokens, _ = preprocess_sentence(sentence)
            features, _ = prepare_data([sentence])
            
            X = self.vectorizer.transform(features)
            predictions = self.model.predict(X)
            
            tagged_tokens = [f"{token}_{pred}" for token, pred in zip(original_tokens, predictions)]
            logger.info("Successfully processed sentence")
            return ' '.join(tagged_tokens), list(zip(original_tokens, predictions))
        
        except Exception as e:
            logger.error(f"Error processing sentence: {str(e)}")
            raise

if __name__ == "__main__":
    model_dir = "model"
    predictor = NERPredictor(model_dir)
    
    sentence = "Washington DC is the capital of United States of America"
    try:
        result, _ = predictor.predict(sentence)
        logger.info(f"Input: {sentence}")
        logger.info(f"Output: {result}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
