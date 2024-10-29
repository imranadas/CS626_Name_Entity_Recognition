# inference.py

import json
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from model_loader import load_model_components
from logger_config import setup_logger
from utils import prepare_data, preprocess_sentence, NamedEntityFeatures

logger = setup_logger('inference')

class NERPredictor:
    """Enhanced NER predictor with parallel processing"""
    
    def __init__(self, model_dir):
        """Initialize the predictor with model and features"""
        logger.info(f"Initializing NER predictor with model directory: {model_dir}")
        
        try:
            self.model, self.vectorizer, self.scaler = load_model_components(model_dir)
            self.ne_features = NamedEntityFeatures()
            self.n_jobs = cpu_count()
            
            # Load feature info and thresholds
            with open(f"{model_dir}/feature_info.json", 'r') as f:
                feature_info = json.load(f)
                self.threshold = feature_info['best_thresholds']['valid']
            
            logger.info(f"Successfully initialized NER predictor with {self.n_jobs} CPU cores")
            
        except Exception as e:
            logger.error(f"Failed to initialize NER predictor: {str(e)}")
            raise
    
    def process_batch(self, sentences):
        """Process a batch of sentences in parallel"""
        def process_single(sentence):
            tokens, preprocessed = preprocess_sentence(sentence)
            # Pass None for labels since we don't have them during inference
            features, _ = prepare_data([sentence], [tokens], None)
            return tokens, features
        
        # Process sentences in parallel
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_single)(sentence) for sentence in sentences
        )
        
        # Unzip results
        tokens_list, features_list = zip(*results)
        all_features = [f for features in features_list for f in features]
        
        return tokens_list, all_features
    
    def _apply_consistency_rules(self, tokens, predictions, scores):
        """Apply post-processing rules for consistent entity tagging"""
        logger.info("Applying consistency rules to predictions")
        try:
            adjusted_predictions = predictions.copy()
            
            for i in range(len(tokens)):
                curr_token = tokens[i].lower()
                curr_score = scores[i]
                
                # Rule 1: Handle administrative units in context
                if curr_token in self.ne_features.admin_units and i > 0:
                    if predictions[i-1] == 1 or (i < len(tokens)-1 and predictions[i+1] == 1):
                        adjusted_predictions[i] = 1
                
                # Rule 2: Handle entity connectors in sequences
                if curr_token in self.ne_features.entity_connectors:
                    if self._is_in_entity_sequence(tokens, predictions, i):
                        adjusted_predictions[i] = 1
                
                # Rule 3: Maintain capitalized sequences
                if self._is_capitalized_sequence(tokens, predictions, i):
                    adjusted_predictions[i] = 1
                
                # Rule 4: High/Low confidence overrides based on LinearSVC decision function
                if curr_score > 2.0:  # High confidence threshold for LinearSVC
                    adjusted_predictions[i] = 1
                elif curr_score < -2.0:  # Low confidence threshold for LinearSVC
                    adjusted_predictions[i] = 0
            
            return adjusted_predictions
            
        except Exception as e:
            logger.error(f"Error applying consistency rules: {str(e)}")
            raise
    
    def _is_in_entity_sequence(self, tokens, predictions, i):
        """Check if token is part of an entity sequence"""
        if i > 0 and i < len(tokens) - 1:
            prev_is_entity = predictions[i-1] == 1
            next_is_entity = predictions[i+1] == 1
            return prev_is_entity and next_is_entity
        return False
    
    def _is_capitalized_sequence(self, tokens, predictions, i):
        """Check if token is part of a capitalized sequence"""
        if i > 0 and i < len(tokens) - 1:
            prev_cap = tokens[i-1][0].isupper()
            curr_cap = tokens[i][0].isupper()
            next_cap = tokens[i+1][0].isupper()
            
            surrounding_pred = (
                (predictions[i-1] if i > 0 else 0) +
                (predictions[i+1] if i < len(tokens)-1 else 0)
            )
            
            return (prev_cap and curr_cap and next_cap and surrounding_pred >= 1)
        return False
    
    def predict_batch(self, sentences, batch_size=32):
        """Predict named entities for multiple sentences using parallel processing"""
        logger.info(f"Processing batch of {len(sentences)} sentences")
        try:
            results = []
            
            # Process sentences in parallel batches
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Parallel processing of batch
                tokens_list, batch_features = self.process_batch(batch)
                
                # Transform features
                X_vectorized = self.vectorizer.transform(batch_features)
                X_scaled = self.scaler.transform(X_vectorized)
                
                # Get predictions
                scores = self.model.decision_function(X_scaled)
                predictions = (scores >= self.threshold).astype(int)
                
                # Process results for each sentence
                start_idx = 0
                for tokens in tokens_list:
                    n_tokens = len(tokens)
                    sent_predictions = predictions[start_idx:start_idx + n_tokens]
                    sent_scores = scores[start_idx:start_idx + n_tokens]
                    
                    # Apply consistency rules
                    adjusted_predictions = self._apply_consistency_rules(
                        tokens, sent_predictions, sent_scores
                    )
                    
                    # Create tagged output
                    tagged_text = ' '.join([
                        f"{token}_{pred}"
                        for token, pred in zip(tokens, adjusted_predictions)
                    ])
                    
                    results.append({
                        'sentence': ' '.join(tokens),
                        'tagged_text': tagged_text,
                        'predictions': list(zip(tokens, adjusted_predictions))
                    })
                    
                    start_idx += n_tokens
            
            logger.info("Successfully processed batch")
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def predict(self, sentence):
        """Predict named entities for a single sentence"""
        results = self.predict_batch([sentence])
        return results[0]['tagged_text'], results[0]['predictions']

if __name__ == "__main__":
    test_sentences = [
        "Washington DC is the capital of United States of America",
        "The United States Department of Defense is in Washington",
        "The University of California is located in Los Angeles",
    ]
    
    try:
        predictor = NERPredictor("model")
        results = predictor.predict_batch(test_sentences)
        
        for result in results:
            logger.info(f"\nInput: {result['sentence']}")
            logger.info(f"Output: {result['tagged_text']}")
            
    except Exception as e:
        logger.error(f"Error in test execution: {str(e)}")
