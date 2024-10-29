# Named Entity Recognition System

A streamlined Named Entity Recognition (NER) system built with Python, featuring a user-friendly web interface for entity detection and model performance visualization.

## Features

- **Named Entity Recognition**: Detect and highlight named entities in input text
- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Model Performance Metrics**: Visualize model performance across different dataset splits
- **Comprehensive Logging**: Detailed logging system for tracking operations and debugging
- **CoNLL-2003 Dataset Integration**: Automatic dataset download and processing
- **Model Training Pipeline**: Complete pipeline for training and evaluating NER models

## Project Structure

```
named-entity-recognition/
├── app.py                 # Streamlit web application
├── data_utils.py         # Dataset handling utilities
├── inference.py          # Model inference functionality
├── logger_config.py      # Logging configuration
├── model_loader.py       # Model loading utilities
├── requirements.txt      # Project dependencies
├── train.py             # Model training script
├── utils.py             # General utility functions
├── logs/                # Log files directory
├── data/                # Dataset directory
│   ├── conll_2003/     # Raw dataset files
│   └── processed_conll_2003/ # Processed dataset files
└── model/               # Trained model files
    ├── model.pkl        # Trained model
    ├── vectorizer.pkl   # Feature vectorizer
    └── metrics.json     # Model performance metrics
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/imranadas/CS626_Name_Entity_Recognition.git
cd named-entity-recognition
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Run the training script:
```bash
python train.py
```
This will:
- Download the CoNLL-2003 dataset if not present
- Process the dataset
- Train the NER model
- Save the model and performance metrics

### Running the Web Interface

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the displayed URL (typically http://localhost:8501)

### Using the Interface

1. **NER Inference Tab**:
   - Enter text in the input area
   - Click "Identify Named Entities"
   - View results with highlighted named entities

2. **Model Metrics Tab**:
   - View performance metrics across different dataset splits
   - Examine confusion matrices
   - Check detailed performance statistics

## Logging

The system maintains comprehensive logs in the `logs` directory:
- Each module has its own log file with timestamp
- Logs include INFO, WARNING, and ERROR level messages
- Both file and console logging are supported

Log files are named in the format: `{module_name}_{timestamp}.log`

## Dependencies

- numpy: Numerical computing
- scikit-learn: Machine learning functionality
- streamlit: Web interface
- pandas: Data manipulation
- plotly: Interactive visualizations
- tqdm: Progress bars

## Model Details

- Algorithm: Linear Support Vector Classification (LinearSVC)
- Features: Token-based features including:
  - Capitalization patterns
  - Token length
  - Surrounding context
  - Prefix/suffix information

## Performance Metrics

The system tracks:
- Precision
- Recall
- F1 Score
- Confusion Matrices

Metrics are calculated for:
- Training set
- Validation set
- Test set

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CoNLL-2003 dataset providers
- Streamlit team for the amazing web framework
- scikit-learn team for machine learning tools

## Contact

Your Name - [@imranadas](https://github.com/imranadas)

Project Link: [https://github.com/imranadas/CS626_Name_Entity_Recognition](https://github.com/imranadas/CS626_Name_Entity_Recognition)