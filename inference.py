# inference.py

import os
import json
import spacy
import numpy as np
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from model_loader import load_model_components
from logger_config import setup_logger
from utils import prepare_data, preprocess_sentence, NamedEntityFeatures

logger = setup_logger('inference')

def get_test_sentences():
    """Get categorized test sentences for NER evaluation from JSON file"""
    try:
        # Try to load from common locations
        test_data_paths = [
            'test_data.json',
            os.path.join('data', 'test_data.json'),
            os.path.join(os.path.dirname(__file__), 'test_data.json'),
            os.path.join(os.path.dirname(__file__), 'data', 'test_data.json')
        ]
        
        test_data_file = None
        for path in test_data_paths:
            if os.path.exists(path):
                test_data_file = path
                break
        
        if test_data_file is None:
            logger.error("test_data.json not found in any of the expected locations")
            return {}, []
        
        logger.info(f"Loading test data from: {test_data_file}")
        with open(test_data_file, 'r', encoding='utf-8') as f:
            test_categories = json.load(f)
        
        # Create flat list of all sentences
        all_sentences = []
        for sentences in test_categories.values():
            all_sentences.extend(sentences)
        
        logger.info(f"Loaded {len(test_categories)} categories with {len(all_sentences)} total sentences")
        return test_categories, all_sentences
        
    except Exception as e:
        logger.error(f"Error loading test data: {str(e)}")
        return {}, []

class POSTagger:
    """POS Tagger wrapper using spaCy"""
    
    def __init__(self):
        """Initialize spaCy model for POS tagging"""
        logger.info("Initializing spaCy model for POS tagging")
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser', 'lemmatizer'])
            logger.info("Successfully loaded spaCy model")
        except Exception as e:
            logger.error(f"Failed to load spaCy model: {str(e)}")
            raise
    
    def tag_sentence(self, tokens):
        """
        Get POS tags for a list of tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: List of Penn Treebank POS tags
        """
        try:
            # Create a fresh text by joining tokens
            text = ' '.join(str(t) for t in tokens)
            doc = self.nlp(text)
            
            # Create mapping of token positions
            token_map = {}
            curr_pos = 0
            for token in tokens:
                token_map[curr_pos] = token
                curr_pos += len(str(token)) + 1  # +1 for space
            
            # Assign tags based on position overlap
            tags = ['UNK'] * len(tokens)
            token_idx = 0
            
            for spacy_token in doc:
                if token_idx < len(tokens):
                    if str(spacy_token) == str(tokens[token_idx]):
                        tags[token_idx] = spacy_token.tag_
                        token_idx += 1
                    else:
                        tags[token_idx] = self._guess_tag(tokens[token_idx])
                        token_idx += 1
            
            return tags
        except Exception as e:
            logger.error(f"Error during POS tagging: {str(e)}")
            return ['UNK'] * len(tokens)
    
    def _guess_tag(self, token):
        """Guess POS tag based on token characteristics"""
        token = str(token)
        if token[0].isupper():
            return 'NNP'  # Proper noun
        if token.isdigit():
            return 'CD'   # Cardinal number
        if token in '.!?':
            return '.'    # Punctuation
        if token == ',':
            return ','    # Comma
        if token.lower() in {'the', 'a', 'an'}:
            return 'DT'   # Determiner
        if token.lower() in {'in', 'on', 'at', 'by', 'with'}:
            return 'IN'   # Preposition
        return 'NN'       # Default to noun

class NERPredictor:
    """Named Entity Recognition predictor with POS tag integration"""
    
    def __init__(self, model_dir):
        """Initialize the predictor with model and features"""
        logger.info(f"Initializing NER predictor with model directory: {model_dir}")
        
        try:
            self.model, self.vectorizer, self.scaler = load_model_components(model_dir)
            self.ne_features = NamedEntityFeatures()
            self.pos_tagger = POSTagger()
            self.n_jobs = cpu_count()
            
            # Load feature info and thresholds
            with open(f"{model_dir}/feature_info.json", 'r') as f:
                feature_info = json.load(f)
                self.threshold = feature_info['best_thresholds']['valid']
            
            logger.info(f"Successfully initialized NER predictor with {self.n_jobs} CPU cores")
            
        except Exception as e:
            logger.error(f"Failed to initialize NER predictor: {str(e)}")
            raise
    
    @staticmethod
    def get_test_cases():
        """Get test cases for NER evaluation"""
        return get_test_sentences()
    
    def process_single(self, sentence):
        """Process a single sentence"""
        try:
            # Get tokens
            tokens, preprocessed = preprocess_sentence(sentence)
            
            # Get POS tags
            pos_tags = self.pos_tagger.tag_sentence(tokens)
            
            # Prepare features
            features, _ = prepare_data([sentence], [tokens], [pos_tags], None)
            
            return tokens, features, pos_tags
        except Exception as e:
            logger.error(f"Error processing sentence '{sentence}': {str(e)}")
            raise
    
    def process_batch(self, sentences):
        """Process a batch of sentences"""
        try:
            results = [self.process_single(sent) for sent in sentences]
            tokens_list, features_list, pos_tags_list = zip(*results)
            all_features = [f for features in features_list for f in features]
            return tokens_list, all_features, pos_tags_list
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
            raise
    
    def predict(self, text):
        """Predict named entities for a single text input"""
        results = self.predict_batch([text])
        return results[0]['tagged_text'], results[0]['predictions']
    
    def predict_batch(self, sentences, batch_size=32):
        """Predict named entities for multiple sentences"""
        logger.info(f"Processing batch of {len(sentences)} sentences")
        try:
            results = []
            
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i + batch_size]
                
                # Process batch
                tokens_list, batch_features, pos_tags_list = self.process_batch(batch)
                
                # Transform features
                X_vectorized = self.vectorizer.transform(batch_features)
                X_scaled = self.scaler.transform(X_vectorized)
                
                # Get predictions
                scores = self.model.decision_function(X_scaled)
                predictions = (scores >= self.threshold).astype(int)
                
                # Process results for each sentence
                start_idx = 0
                for tokens, pos_tags in zip(tokens_list, pos_tags_list):
                    n_tokens = len(tokens)
                    sent_predictions = predictions[start_idx:start_idx + n_tokens]
                    sent_scores = scores[start_idx:start_idx + n_tokens]
                    
                    # Apply consistency rules
                    adjusted_predictions = self._apply_consistency_rules(
                        tokens, sent_predictions, sent_scores, pos_tags
                    )
                    
                    results.append({
                        'sentence': ' '.join(tokens),
                        'tagged_text': self._format_tagged_text(tokens, adjusted_predictions),
                        'predictions': list(zip(tokens, adjusted_predictions)),
                        'pos_tags': pos_tags
                    })
                    
                    start_idx += n_tokens
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise
    
    def _apply_consistency_rules(self, tokens, predictions, scores, pos_tags):
        """Apply post-processing rules for consistent entity tagging"""
        adjusted_predictions = predictions.copy()
        
        # First pass: Handle proper noun sequences
        sequence_start = None
        
        for i in range(len(tokens)):
            curr_token = tokens[i].lower()
            curr_pos = pos_tags[i]
            
            # Start or continue sequence
            if (curr_pos.startswith('NNP') or 
                curr_token in {'prime', 'minister', 'president', 'chief'} or
                (i > 0 and tokens[i-1].lower() in {'prime', 'chief'} and 
                 curr_token in {'minister', 'justice'})):
                if sequence_start is None:
                    sequence_start = i
            # End sequence
            elif sequence_start is not None:
                sequence = tokens[sequence_start:i]
                if self._is_likely_entity(sequence):
                    for j in range(sequence_start, i):
                        adjusted_predictions[j] = 1
                sequence_start = None
        
        # Handle last sequence
        if sequence_start is not None:
            sequence = tokens[sequence_start:]
            if self._is_likely_entity(sequence):
                for j in range(sequence_start, len(tokens)):
                    adjusted_predictions[j] = 1
        
        # Second pass: Apply confidence scores and cleanup
        for i in range(len(tokens)):
            curr_score = scores[i]
            
            # High confidence overrides
            if curr_score > 1.8:
                adjusted_predictions[i] = 1
            elif curr_score < -2.0:
                adjusted_predictions[i] = 0
        
        # Third pass: Clean up isolated predictions
        for i in range(1, len(tokens) - 1):
            if (adjusted_predictions[i] == 1 and
                adjusted_predictions[i-1] == 0 and
                adjusted_predictions[i+1] == 0 and
                scores[i] < 1.5):
                adjusted_predictions[i] = 0
        
        return adjusted_predictions
    
    def _is_likely_entity(self, tokens):
        """Check if a sequence of tokens is likely to be a named entity"""
        lower_tokens = [t.lower() for t in tokens]
        
        # Common organization endings
        org_endings = {'university', 'institute', 'college', 'school', 'corporation', 
                      'inc', 'ltd', 'limited', 'corp', 'department', 'ministry', 
                      'organization', 'committee', 'council'}
        
        # Common person titles and designations
        person_titles = {
            'mr', 'mrs', 'ms', 'dr', 'prof', 'sir', 'lady',  # Basic titles
            'president', 'prime', 'minister', 'chancellor',    # Political titles
            'ceo', 'director', 'chairman', 'chairperson',     # Corporate titles
            'king', 'queen', 'prince', 'princess',            # Royal titles
            'judge', 'justice', 'chief'                       # Legal titles
        }
        
        # Check for title combinations
        title_pairs = {
            ('prime', 'minister'),
            ('chief', 'minister'),
            ('vice', 'president'),
            ('deputy', 'director')
        }
        
        # Check various entity patterns
        is_org = any(token in org_endings for token in lower_tokens)
        
        # Check for title pairs
        has_title_pair = any(
            lower_tokens[i:i+2] == list(pair)
            for i in range(len(lower_tokens)-1)
            for pair in title_pairs
        )
        
        # Check for titles
        is_title = (
            any(token in person_titles for token in lower_tokens) or
            has_title_pair
        )
        
        # Check if all tokens are capitalized
        all_caps = all(t[0].isupper() for t in tokens)
        
        # Check if it contains common entity connectors
        has_connector = any(token in self.ne_features.entity_connectors for token in lower_tokens)
        
        return (is_org or is_title or (all_caps and has_connector))
    
    def _format_tagged_text(self, tokens, predictions):
        """Format tagged text with entity brackets"""
        tagged_tokens = []
        in_entity = False
        
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            if pred == 1 and not in_entity:
                tagged_tokens.append('[')
                in_entity = True
            elif pred == 0 and in_entity:
                tagged_tokens.append(']')
                in_entity = False
            
            tagged_tokens.append(token)
            
            # Close entity at sentence end if needed
            if in_entity and i == len(tokens) - 1:
                tagged_tokens.append(']')
        
        return ' '.join(tagged_tokens)

if __name__ == "__main__":
    # Test the predictor
    test_categories, all_sentences = get_test_sentences()
    
    if not test_categories:
        logger.error("No test data available. Please ensure test_data.json is present.")
    else:
        try:
            predictor = NERPredictor("model")
            
            logger.info(f"\nProcessing {len(test_categories)} categories of test data:")
            logger.info("=" * 80)
            
            # Process each category
            for category, sentences in test_categories.items():
                logger.info(f"\nCategory: {category}")
                logger.info("-" * 40)
                
                results = predictor.predict_batch(sentences)
                
                # Process results for this category
                for idx, result in enumerate(results, 1):
                    logger.info(f"\nExample {idx}:")
                    logger.info(f"Input: {result['sentence']}")
                    logger.info(f"Tagged: {result['tagged_text']}")
                    
                    # Extract and display entities
                    entities = [token for token, pred in result['predictions'] if pred == 1]
                    if entities:
                        logger.info(f"Entities: {', '.join(entities)}")
                    
                    # Display POS tags for tokens
                    pos_info = [f"{token}({tag})" for token, tag in zip(result['sentence'].split(), result['pos_tags'])]
                    logger.info(f"POS Tags: {' '.join(pos_info)}")
                
                # Calculate category statistics
                total_entities = sum(len([token for token, pred in result['predictions'] if pred == 1]) 
                                   for result in results)
                avg_entities = total_entities / len(results)
                logger.info(f"\nCategory Statistics:")
                logger.info(f"Total Entities: {total_entities}")
                logger.info(f"Average Entities per Sentence: {avg_entities:.2f}")
                logger.info("=" * 80)
            
            # Overall statistics
            all_results = predictor.predict_batch(all_sentences)
            total_entities = sum(len([token for token, pred in result['predictions'] if pred == 1]) 
                               for result in all_results)
            total_sentences = len(all_sentences)
            
            logger.info("\nOverall Statistics:")
            logger.info("=" * 40)
            logger.info(f"Total Categories: {len(test_categories)}")
            logger.info(f"Total Test Sentences: {total_sentences}")
            logger.info(f"Total Entities Identified: {total_entities}")
            logger.info(f"Average Entities per Sentence: {total_entities/total_sentences:.2f}")
            logger.info("=" * 40)
            
        except Exception as e:
            logger.error(f"Error in test execution: {str(e)}")
            raise
