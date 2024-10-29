# app.py

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from inference import NERPredictor
from logger_config import setup_logger

# Initialize logger
logger = setup_logger('streamlit_app')

# Set page config
st.set_page_config(
    page_title="Named Entity Recognition System",
    page_icon="🏷️",
    layout="wide"
)

def load_metrics(model_dir):
    """
    Load model metrics from JSON file
    
    Args:
        model_dir (str): Directory containing metrics file
        
    Returns:
        dict: Metrics dictionary or None if file not found
    """
    logger.info(f"Loading metrics from: {model_dir}")
    metrics_path = os.path.join(model_dir, 'metrics.json')
    
    try:
        if not os.path.exists(metrics_path):
            logger.warning("Metrics file not found")
            return None
            
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            logger.info("Successfully loaded metrics")
            return metrics
    except Exception as e:
        logger.error(f"Error loading metrics: {str(e)}")
        return None

def plot_metrics_comparison(metrics):
    """
    Create a bar plot comparing metrics across splits
    
    Args:
        metrics (dict): Dictionary containing metrics for each split
        
    Returns:
        plotly.graph_objects.Figure: Comparison plot
    """
    logger.info("Generating metrics comparison plot")
    try:
        # Prepare data for plotting
        data = []
        for split in metrics.keys():
            for metric in ['precision', 'recall', 'f1']:
                data.append({
                    'Split': split.capitalize(),
                    'Metric': metric.capitalize(),
                    'Value': metrics[split][metric]
                })
        
        df = pd.DataFrame(data)
        
        fig = px.bar(
            df,
            x='Split',
            y='Value',
            color='Metric',
            barmode='group',
            title='Model Performance Metrics Across Splits',
            labels={'Value': 'Score', 'Split': 'Dataset Split'},
            text=df['Value'].apply(lambda x: f'{x:.3f}')
        )
        
        fig.update_layout(
            yaxis_range=[0, 1],
            plot_bgcolor='white',
            yaxis_gridcolor='lightgray'
        )
        
        logger.info("Successfully generated metrics comparison plot")
        return fig
    except Exception as e:
        logger.error(f"Error generating metrics comparison plot: {str(e)}")
        raise

def plot_confusion_matrix(conf_matrix, split_name):
    """
    Create a heatmap of the confusion matrix
    
    Args:
        conf_matrix (list): 2D list containing confusion matrix values
        split_name (str): Name of the dataset split
        
    Returns:
        plotly.graph_objects.Figure: Confusion matrix heatmap
    """
    logger.info(f"Generating confusion matrix plot for {split_name} split")
    try:
        labels = ['Non-Entity', 'Named Entity']
        
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=labels,
            y=labels,
            text=[[str(int(val)) for val in row] for row in conf_matrix],
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False,
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {split_name} Split',
            xaxis_title='Predicted',
            yaxis_title='Actual',
            width=400,
            height=400
        )
        
        logger.info(f"Successfully generated confusion matrix for {split_name} split")
        return fig
    except Exception as e:
        logger.error(f"Error generating confusion matrix plot: {str(e)}")
        raise

def display_tagged_text(tokens_with_predictions):
    """
    Display color-coded text based on predictions
    
    Args:
        tokens_with_predictions (list): List of (token, prediction) pairs
    """
    logger.info("Displaying tagged text")
    try:
        html = []
        for token, pred in tokens_with_predictions:
            if pred == 1:
                html.append(f'<span style="background-color: #90EE90">{token}</span>')
            else:
                html.append(token)
        
        st.markdown(' '.join(html), unsafe_allow_html=True)
        logger.info("Successfully displayed tagged text")
    except Exception as e:
        logger.error(f"Error displaying tagged text: {str(e)}")
        raise

def main():
    """Main function for the Streamlit application"""
    logger.info("Starting Named Entity Recognition System")
    
    st.title("Named Entity Recognition System 🏷️")
    
    # Initialize predictor
    try:
        predictor = NERPredictor("model")
        model_loaded = True
        logger.info("Successfully initialized NER predictor")
    except FileNotFoundError:
        error_msg = "Model files not found. Please train the model first."
        logger.error(error_msg)
        st.error(error_msg)
        model_loaded = False
    
    # Create tabs
    tab1, tab2 = st.tabs(["NER Inference", "Model Metrics"])
    
    # Tab 1: NER Inference
    with tab1:
        logger.info("Rendering NER Inference tab")
        st.header("Named Entity Recognition")
        st.write("Enter a sentence to identify named entities.")
        
        # Text input
        input_text = st.text_area(
            "Input Text",
            value="Washington DC is the capital of United States of America",
            height=100
        )
        
        if st.button("Identify Named Entities") and model_loaded:
            logger.info(f"Processing input text: {input_text}")
            with st.spinner("Processing..."):
                try:
                    # Get predictions
                    _, tokens_with_predictions = predictor.predict(input_text)
                    
                    # Display results
                    st.subheader("Results")
                    st.write("Named entities are highlighted in green:")
                    display_tagged_text(tokens_with_predictions)
                    
                    # Display legend
                    st.markdown("---")
                    st.markdown("""
                    **Legend:**
                    - <span style='background-color: #90EE90'>Highlighted text</span>: Named Entity
                    - Regular text: Non-Entity
                    """, unsafe_allow_html=True)
                    
                    logger.info("Successfully processed and displayed results")
                except Exception as e:
                    error_msg = f"Error processing text: {str(e)}"
                    logger.error(error_msg)
                    st.error(error_msg)
    
    # Tab 2: Model Metrics
    with tab2:
        logger.info("Rendering Model Metrics tab")
        st.header("Model Performance Metrics")
        
        metrics = load_metrics("model")
        if metrics:
            try:
                # Display overall metrics
                st.subheader("Performance Overview")
                metrics_fig = plot_metrics_comparison(metrics)
                st.plotly_chart(metrics_fig, use_container_width=True)
                
                # Display confusion matrices
                st.subheader("Confusion Matrices")
                cols = st.columns(3)
                
                for idx, (split_name, split_metrics) in enumerate(metrics.items()):
                    with cols[idx]:
                        st.write(f"**{split_name.capitalize()} Split**")
                        conf_matrix_fig = plot_confusion_matrix(
                            split_metrics['confusion_matrix'],
                            split_name.capitalize()
                        )
                        st.plotly_chart(conf_matrix_fig)
                
                # Display detailed metrics table
                st.subheader("Detailed Metrics")
                detailed_metrics = []
                for split_name, split_metrics in metrics.items():
                    row = {
                        'Split': split_name.capitalize(),
                        'Precision': f"{split_metrics['precision']:.3f}",
                        'Recall': f"{split_metrics['recall']:.3f}",
                        'F1 Score': f"{split_metrics['f1']:.3f}"
                    }
                    detailed_metrics.append(row)
                
                st.table(pd.DataFrame(detailed_metrics))
                logger.info("Successfully displayed model metrics")
                
            except Exception as e:
                error_msg = f"Error displaying metrics: {str(e)}"
                logger.error(error_msg)
                st.error(error_msg)
        else:
            logger.warning("No metrics found")
            st.error("No metrics found. Please train the model first.")

if __name__ == "__main__":
    main()
