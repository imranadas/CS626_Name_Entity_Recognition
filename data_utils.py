# data_utils.py

import os
import shutil
import zipfile
import urllib.request
from tqdm import tqdm
from logger_config import setup_logger

logger = setup_logger('data_utils')

class DownloadProgressBar(tqdm):
    """Custom progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

class CoNLLDatasetLoader:
    """Class to handle downloading and processing of CoNLL-2003 dataset"""
    
    def __init__(self, data_dir="data"):
        """
        Initialize the dataset loader
        
        Args:
            data_dir (str): Base directory for data storage
        """
        self.data_dir = data_dir
        self.dataset_url = "https://data.deepai.org/conll2003.zip"
        self.zip_path = os.path.join(data_dir, "conll2003.zip")
        self.extract_dir = os.path.join(data_dir, "conll_2003")
        self.processed_dir = os.path.join(data_dir, "processed_conll_2003")
        logger.info(f"Initialized CoNLLDatasetLoader with data directory: {data_dir}")
    
    def download_and_extract(self):
        """
        Download and extract the CoNLL-2003 dataset with progress bar
        
        Raises:
            Exception: For download or extraction errors
        """
        logger.info("Starting dataset download and extraction")
        os.makedirs(self.data_dir, exist_ok=True)
        
        try:
            logger.info("Downloading CoNLL-2003 dataset...")
            with DownloadProgressBar(unit='B', unit_scale=True,
                                   miniters=1, desc="Downloading") as t:
                urllib.request.urlretrieve(self.dataset_url, self.zip_path,
                                         reporthook=t.update_to)
            
            logger.info("Extracting dataset...")
            os.makedirs(self.extract_dir, exist_ok=True)
            
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                for file in file_list:
                    if file.endswith('.txt'):
                        base_name = os.path.basename(file)
                        if 'train' in file.lower():
                            output_name = 'train.txt'
                        elif 'valid' in file.lower() or 'dev' in file.lower():
                            output_name = 'valid.txt'
                        elif 'test' in file.lower():
                            output_name = 'test.txt'
                        else:
                            continue
                        
                        logger.info(f"Extracting {file} as {output_name}")
                        with zip_ref.open(file) as source, \
                             open(os.path.join(self.extract_dir, output_name), 'wb') as target:
                            shutil.copyfileobj(source, target)
            
            os.remove(self.zip_path)
            logger.info("Dataset downloaded and extracted successfully!")
            
        except Exception as e:
            logger.error(f"Error downloading or extracting dataset: {str(e)}")
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)
            if os.path.exists(self.extract_dir):
                shutil.rmtree(self.extract_dir)
            raise
    
    def process_file(self, input_path, output_path):
        """
        Process CoNLL format file and convert to simplified format with POS tags
        
        Args:
            input_path (str): Path to input file
            output_path (str): Path to output file
        """
        logger.info(f"Processing file: {input_path}")
        try:
            with open(input_path, 'r', encoding='utf-8') as f_in, \
                open(output_path, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    line = line.strip()
                    if line.startswith("-DOCSTART-") or not line:
                        f_out.write("\n")
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 4:
                        token = parts[0]
                        pos_tag = parts[1]  # Extract POS tag
                        ner_tag = parts[3]
                        f_out.write(f"{token}\t{pos_tag}\t{ner_tag}\n")  # Tab-separated format
            
            logger.info(f"Successfully processed file: {input_path}")
        except Exception as e:
            logger.error(f"Error processing file {input_path}: {str(e)}")
            raise
    
    def verify_dataset_files(self, directory):
        """
        Verify that all required dataset files exist and are non-empty
        
        Args:
            directory (str): Directory to check for files
            
        Returns:
            list: List of missing or empty files
        """
        logger.info(f"Verifying dataset files in {directory}")
        required_files = ['train.txt', 'valid.txt', 'test.txt']
        missing_files = []
        
        for file_name in required_files:
            file_path = os.path.join(directory, file_name)
            if not os.path.exists(file_path):
                missing_files.append(file_name)
            else:
                if os.path.getsize(file_path) == 0:
                    missing_files.append(f"{file_name} (empty)")
        
        if missing_files:
            logger.warning(f"Missing or empty files: {', '.join(missing_files)}")
        else:
            logger.info("All required files present and non-empty")
        
        return missing_files
    
    def prepare_dataset(self, force_download=False):
        """
        Prepare the dataset by downloading and processing all splits
        
        Args:
            force_download (bool): Force re-download even if files exist
            
        Returns:
            str: Path to processed directory
        """
        logger.info("Starting dataset preparation")
        try:
            if force_download and os.path.exists(self.extract_dir):
                logger.info("Removing existing extracted data due to force_download")
                shutil.rmtree(self.extract_dir)
            
            if not os.path.exists(self.extract_dir) or force_download:
                self.download_and_extract()
            
            missing_files = self.verify_dataset_files(self.extract_dir)
            if missing_files:
                logger.error(f"Missing required dataset files: {', '.join(missing_files)}")
                raise FileNotFoundError(f"Missing required dataset files: {', '.join(missing_files)}")
            
            os.makedirs(self.processed_dir, exist_ok=True)
            
            splits = {
                'train': ('train.txt', 'train.txt'),
                'valid': ('valid.txt', 'valid.txt'),
                'test': ('test.txt', 'test.txt')
            }
            
            for split_name, (input_file, output_file) in splits.items():
                logger.info(f"Processing {split_name} split...")
                input_path = os.path.join(self.extract_dir, input_file)
                output_path = os.path.join(self.processed_dir, output_file)
                self.process_file(input_path, output_path)
            
            logger.info("Dataset preparation completed successfully!")
            return self.processed_dir
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            if os.path.exists(self.zip_path):
                os.remove(self.zip_path)
            raise
    
    def prepare_existing_dataset(self, conll_dir):
        """
        Process an existing dataset without downloading
        
        Args:
            conll_dir (str): Directory containing existing CoNLL data
            
        Returns:
            str: Path to processed directory
        """
        logger.info(f"Preparing existing dataset from: {conll_dir}")
        try:
            if not os.path.exists(conll_dir):
                error_msg = f"Directory not found: {conll_dir}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            missing_files = self.verify_dataset_files(conll_dir)
            if missing_files:
                error_msg = f"Missing required dataset files in {conll_dir}: {', '.join(missing_files)}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            os.makedirs(self.processed_dir, exist_ok=True)
            
            splits = {
                'train': ('train.txt', 'train.txt'),
                'valid': ('valid.txt', 'valid.txt'),
                'test': ('test.txt', 'test.txt')
            }
            
            for split_name, (input_file, output_file) in splits.items():
                logger.info(f"Processing {split_name} split...")
                input_path = os.path.join(conll_dir, input_file)
                output_path = os.path.join(self.processed_dir, output_file)
                self.process_file(input_path, output_path)
            
            logger.info("Existing dataset preparation completed successfully!")
            return self.processed_dir
            
        except Exception as e:
            logger.error(f"Error preparing existing dataset: {str(e)}")
            if os.path.exists(self.processed_dir):
                shutil.rmtree(self.processed_dir)
            raise
