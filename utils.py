# utils.py

import json
import re
import nltk
import string
from pathlib import Path
from nltk.corpus import wordnet as wn
from logger_config import setup_logger

logger = setup_logger('utils')

class NamedEntityFeatures:
    """Manages dynamic named entity feature generation and caching"""
    
    def __init__(self, resource_dir="data"):
        self.resource_dir = Path(resource_dir)
        self._ensure_resources()
        self._load_or_generate_features()
    
    def _ensure_resources(self):
        """Ensure required NLTK resources are available"""
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
    
    def _load_or_generate_features(self):
        """Load features from cache or generate them"""
        cache_file = self.resource_dir / "entity_features.json"
        
        if cache_file.exists():
            logger.info("Loading entity features from cache")
            with open(cache_file, 'r') as f:
                features = json.load(f)
                self.admin_units = set(features['admin_units'])
                self.entity_connectors = set(features['entity_connectors'])
                self.directions = set(features['directions'])
        else:
            logger.info("Generating entity features")
            self._generate_features()
            self._save_features(cache_file)
    
    def _generate_features(self):
        """Generate features using WordNet"""
        self.admin_units = self._get_administrative_units()
        self.entity_connectors = self._get_entity_connectors()
        self.directions = self._get_directions()
    
    def _get_administrative_units(self):
        """Extract administrative units from WordNet"""
        admin_units = set()
        
        admin_synsets = wn.synsets('administrative_district', pos=wn.NOUN)
        admin_synsets.extend(wn.synsets('territory', pos=wn.NOUN))
        admin_synsets.extend(wn.synsets('political_unit', pos=wn.NOUN))
        
        for synset in admin_synsets:
            for lemma in synset.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                admin_units.add(name)
            
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    admin_units.add(name)
        
        # Add common abbreviations and variants
        common_variants = {
            'dc', 'd.c.', 'district of columbia', 'fed', 'federal',
            'natl', 'national', 'dept', 'department'
        }
        admin_units.update(common_variants)
        
        return admin_units
    
    def _get_entity_connectors(self):
        """Generate entity connectors"""
        connectors = {
            'of', 'and', 'the', 'de', 'del', 'van', 'der', 'al',
            'bin', 'ibn', 'von', 'das', 'los', 'las', 'el', 'la',
            '&', '+'
        }
        
        for synset in wn.synsets('conjunction', pos=wn.NOUN):
            for lemma in synset.lemmas():
                name = lemma.name().lower().replace('_', ' ')
                if len(name) < 5:
                    connectors.add(name)
        
        return connectors
    
    def _get_directions(self):
        """Generate directional terms"""
        directions = set()
        base_dirs = {'north', 'south', 'east', 'west'}
        
        for direction in base_dirs:
            directions.add(direction)
            
            synsets = wn.synsets(direction, pos=wn.NOUN)
            for synset in synsets:
                for lemma in synset.lemmas():
                    name = lemma.name().lower().replace('_', ' ')
                    directions.add(name)
                    
                    if name.endswith('ern'):
                        directions.add(name[:-3] + 'ern')
                    directions.add(name + 'ern')
                    directions.add(name + 'erly')
        
        compounds = {
            'northeast', 'northwest', 'southeast', 'southwest',
            'north-east', 'north-west', 'south-east', 'south-west',
            'northeastern', 'northwestern', 'southeastern', 'southwestern'
        }
        directions.update(compounds)
        
        return directions
    
    def _save_features(self, cache_file):
        """Save features to cache"""
        self.resource_dir.mkdir(parents=True, exist_ok=True)
        
        features = {
            'admin_units': list(self.admin_units),
            'entity_connectors': list(self.entity_connectors),
            'directions': list(self.directions)
        }
        
        with open(cache_file, 'w') as f:
            json.dump(features, f, indent=2)
        
        logger.info("Saved features to cache")
    
    def update_features(self, new_terms):
        """Add new terms to feature sets"""
        for term, feature_type in new_terms.items():
            term = term.lower()
            if feature_type == 'admin':
                self.admin_units.add(term)
            elif feature_type == 'connector':
                self.entity_connectors.add(term)
            elif feature_type == 'direction':
                self.directions.add(term)
        
        cache_file = self.resource_dir / "entity_features.json"
        self._save_features(cache_file)
    
    def get_all_features(self):
        """
        Get all feature sets used for entity recognition
        
        Returns:
            dict: Dictionary containing all feature sets
        """
        return {
            'admin_units': list(self.admin_units),
            'entity_connectors': list(self.entity_connectors),
            'directions': list(self.directions)
        }

def preprocess_sentence(sentence, original_tokens=None):
    """
    Preprocess sentence while maintaining original tokenization if provided
    
    Args:
        sentence (str): Input sentence
        original_tokens (list, optional): Original tokens from CoNLL format
    """
    logger.info(f"Preprocessing sentence: {sentence}")
    try:
        if original_tokens is None:
            # Handle periods in abbreviations
            sentence = re.sub(r'(?<!Mr)(?<!Ms)(?<!Dr)(?<!Jr)\.\s', ' . ', sentence)
            sentence = re.sub(r'([A-Z]\.)(?=[A-Z]\.)', r'\1 ', sentence)
            original_tokens = sentence.split()
        
        preprocessed = ' '.join(original_tokens).lower()
        preprocessed_tokens = preprocessed.split()
        
        if len(original_tokens) != len(preprocessed_tokens):
            logger.error(f"Token length mismatch: {len(original_tokens)} != {len(preprocessed_tokens)}")
            logger.error(f"Original: {original_tokens}")
            logger.error(f"Preprocessed: {preprocessed_tokens}")
            raise ValueError("Tokenization mismatch")
        
        return original_tokens, preprocessed_tokens
    except Exception as e:
        logger.error(f"Error preprocessing sentence: {str(e)}")
        raise

def extract_features(tokens, i, preprocessed_tokens, pos_tags, ne_features):
    """Extract comprehensive features for NER including POS tags"""
    try:
        features = {}
        token = tokens[i]
        preprocessed_token = preprocessed_tokens[i]
        pos_tag = pos_tags[i]
        
        # Basic features
        features.update(_get_basic_features(token, preprocessed_token))
        
        # POS tag features
        features.update(_get_pos_features(pos_tags, i))
        
        # Position and context features
        features.update(_get_position_features(tokens, preprocessed_tokens, i, pos_tags))
        
        # Entity pattern features
        features.update(_get_entity_pattern_features(tokens, preprocessed_tokens, i, ne_features))
        
        # Shape features
        features.update(_get_shape_features(token))
        
        # N-gram features
        features.update(_get_ngram_features(preprocessed_token))
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise
    
def _get_pos_features(pos_tags, i):
    """Extract POS tag-related features"""
    features = {
        'pos': pos_tags[i],
        'is_noun': pos_tags[i].startswith('NN'),
        'is_proper_noun': pos_tags[i] in {'NNP', 'NNPS'},
        'is_verb': pos_tags[i].startswith('VB'),
        'is_adjective': pos_tags[i].startswith('JJ'),
        'is_adverb': pos_tags[i].startswith('RB'),
        'is_determiner': pos_tags[i].startswith('DT'),
        'is_preposition': pos_tags[i].startswith('IN'),
    }
    
    # POS tag sequence features
    if i > 0:
        features['prev_pos'] = pos_tags[i-1]
        features['prev_is_proper_noun'] = pos_tags[i-1] in {'NNP', 'NNPS'}
        if i > 1:
            features['prev_prev_pos'] = pos_tags[i-2]
    
    if i < len(pos_tags) - 1:
        features['next_pos'] = pos_tags[i+1]
        features['next_is_proper_noun'] = pos_tags[i+1] in {'NNP', 'NNPS'}
        if i < len(pos_tags) - 2:
            features['next_next_pos'] = pos_tags[i+2]
    
    # Add POS bigram features
    if i > 0:
        features['pos_bigram_prev'] = f"{pos_tags[i-1]}_{pos_tags[i]}"
    if i < len(pos_tags) - 1:
        features['pos_bigram_next'] = f"{pos_tags[i]}_{pos_tags[i+1]}"
    
    return features

def _get_basic_features(token, preprocessed_token):
    """Extract basic token features"""
    return {
        'word': preprocessed_token,
        'is_capitalized': token[0].isupper(),
        'is_all_caps': token.isupper(),
        'has_caps_inside': any(c.isupper() for c in token[1:]),
        'is_alphanumeric': bool(re.match(r'^(?=.*[a-zA-Z])(?=.*\d).+$', token)),
        'has_punctuation': any(c in string.punctuation for c in token),
        'token_length': len(token),
        'is_short': len(token) <= 3,
        'starts_with_cap': token[0].isupper() if token else False,
        'ends_with_period': token.endswith('.'),
        'has_roman_numeral': bool(re.match(r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$', token.upper())),
    }

def _get_position_features(tokens, preprocessed_tokens, i, pos_tags=None):
    """Extract position and context features"""
    features = {
        'is_first_token': i == 0,
        'is_last_token': i == len(tokens) - 1,
        'is_first_cap_sequence': i > 0 and not tokens[i-1][0].isupper() and tokens[i][0].isupper(),
    }
    
    # Previous token features
    if i > 0:
        features.update({
            'prev_token': preprocessed_tokens[i-1],
            'prev_is_cap': tokens[i-1][0].isupper(),
            'prev_is_period': tokens[i-1] == '.',
            'prev_length': len(tokens[i-1])
        })
        if pos_tags:
            features['prev_pos'] = pos_tags[i-1]
            features['prev_is_proper_noun'] = pos_tags[i-1] in {'NNP', 'NNPS'}
            
        if i > 1:
            features['prev_prev_token'] = preprocessed_tokens[i-2]
            if pos_tags:
                features['prev_prev_pos'] = pos_tags[i-2]
    
    # Next token features
    if i < len(tokens) - 1:
        features.update({
            'next_token': preprocessed_tokens[i+1],
            'next_is_cap': tokens[i+1][0].isupper(),
            'next_is_period': tokens[i+1] == '.',
            'next_length': len(tokens[i+1])
        })
        if pos_tags:
            features['next_pos'] = pos_tags[i+1]
            features['next_is_proper_noun'] = pos_tags[i+1] in {'NNP', 'NNPS'}
            
        if i < len(tokens) - 2:
            features['next_next_token'] = preprocessed_tokens[i+2]
            if pos_tags:
                features['next_next_pos'] = pos_tags[i+2]
    
    return features

def _get_entity_pattern_features(tokens, preprocessed_tokens, i, ne_features):
    """Extract entity pattern features"""
    features = {}
    curr_token_lower = preprocessed_tokens[i].lower()
    
    features['is_admin_unit'] = curr_token_lower in ne_features.admin_units
    features['is_connector'] = curr_token_lower in ne_features.entity_connectors
    features['is_direction'] = curr_token_lower in ne_features.directions
    
    if i > 0 and i < len(tokens) - 1:
        prev_cap = tokens[i-1][0].isupper()
        curr_cap = tokens[i][0].isupper()
        next_cap = tokens[i+1][0].isupper()
        
        features.update({
            'in_cap_sequence': prev_cap and curr_cap,
            'starts_cap_sequence': not prev_cap and curr_cap and next_cap,
            'in_entity_pattern': (prev_cap and curr_cap and next_cap) or
                               (prev_cap and curr_token_lower in ne_features.entity_connectors and next_cap)
        })
    
    return features

def _get_shape_features(token):
    """Extract shape-based features"""
    shape = ''
    last_char_type = None
    
    for char in token:
        if char.isupper():
            char_type = 'X'
        elif char.islower():
            char_type = 'x'
        elif char.isdigit():
            char_type = 'd'
        else:
            char_type = char
        
        if char_type != last_char_type:
            shape += char_type
            last_char_type = char_type
    
    return {
        'shape': shape,
        'shaped_condensed': shape,
        'has_number': 'd' in shape,
        'mixed_case': 'x' in shape and 'X' in shape,
    }

def _get_ngram_features(token):
    """Extract character n-gram features"""
    features = {}
    
    for n in range(1, 4):
        if len(token) >= n:
            features[f'prefix_{n}'] = token[:n]
            features[f'suffix_{n}'] = token[-n:]
    
    return features

def prepare_data(sentences, tokens=None, pos_tags=None, labels=None):
    """Prepare features maintaining original tokenization if available"""
    logger.info(f"Preparing features for {len(sentences)} sentences")
    try:
        all_features = []
        all_labels = []
        ne_features = NamedEntityFeatures()
        
        for idx, sentence in enumerate(sentences):
            # Get tokens if not provided
            if tokens is None:
                curr_tokens, preprocessed_tokens = preprocess_sentence(sentence)
                curr_pos_tags = ['UNK'] * len(curr_tokens)  # Default POS tag
            else:
                curr_tokens = tokens[idx]
                curr_pos_tags = pos_tags[idx] if pos_tags else ['UNK'] * len(curr_tokens)
                _, preprocessed_tokens = preprocess_sentence(sentence, curr_tokens)
            
            # Extract features for each token
            sentence_features = []
            for i in range(len(curr_tokens)):
                features = extract_features(
                    curr_tokens, i, preprocessed_tokens, curr_pos_tags, ne_features)
                sentence_features.append(features)
                
                if labels is not None:
                    all_labels.append(labels[idx][i])
            
            all_features.extend(sentence_features)
        
        if labels is not None and len(all_features) != len(all_labels):
            raise ValueError(f"Feature-label mismatch: {len(all_features)} features, {len(all_labels)} labels")
        
        logger.info(f"Successfully prepared {len(all_features)} feature sets")
        return all_features, all_labels if labels is not None else None
    
    except Exception as e:
        logger.error(f"Error preparing data: {str(e)}")
        raise

def load_conll_data(file_path):
    """Load and process CoNLL format data with POS tags"""
    logger.info(f"Loading CoNLL data from: {file_path}")
    all_sentences = []
    all_tokens = []
    all_pos_tags = []
    all_labels = []
    current_sentence = []
    current_tokens = []
    current_pos_tags = []
    current_labels = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    token, pos_tag, tag = line.split('\t')
                    current_tokens.append(token)
                    current_pos_tags.append(pos_tag)
                    current_sentence.append(token)
                    current_labels.append(1 if tag.startswith(('B-', 'I-')) else 0)
                elif current_sentence:  # End of sentence
                    all_sentences.append(' '.join(current_sentence))
                    all_tokens.append(current_tokens)
                    all_pos_tags.append(current_pos_tags)
                    all_labels.append(current_labels)
                    current_sentence = []
                    current_tokens = []
                    current_pos_tags = []
                    current_labels = []
        
        # Handle last sentence if exists
        if current_sentence:
            all_sentences.append(' '.join(current_sentence))
            all_tokens.append(current_tokens)
            all_pos_tags.append(current_pos_tags)
            all_labels.append(current_labels)
        
        logger.info(f"Loaded {len(all_sentences)} sentences")
        return all_sentences, all_tokens, all_pos_tags, all_labels
    
    except Exception as e:
        logger.error(f"Error loading CoNLL data: {str(e)}")
        raise
