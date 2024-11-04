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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Theme colors */
    :root {
        --primary-bg: rgb(17, 19, 23);
        --secondary-bg: rgb(25, 28, 34);
        --border-color: rgba(250, 250, 250, 0.2);
        --highlight-color: rgba(76, 175, 80, 0.3);
        --text-color: rgb(250, 250, 250);
    }

    /* Section header */
    .section-header {
        color: var(--text-color);
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid var(--border-color);
    }

    /* Entity text container */
    .entity-text {
        font-family: "Source Code Pro", monospace;
        font-size: 0.875rem;
        line-height: 1.6;
        color: var(--text-color);
        background-color: var(--primary-bg);
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
    }

    /* Entity highlight */
    .entity-highlight {
        background-color: var(--highlight-color);
        color: var(--text-color);
        padding: 0.125rem 0.25rem;
        border-radius: 0.25rem;
        margin: 0 0.125rem;
        border: 1px solid rgba(76, 175, 80, 0.5);
    }

    /* POS tag styling */
    .pos-tag {
        color: rgba(250, 250, 250, 0.6);
        font-size: 0.7em;
        position: relative;
        top: -0.5em;
        margin-left: 0.125rem;
    }

    /* Entity summary section */
    .entities-box {
        background-color: var(--primary-bg);
        border: 1px solid var(--border-color);
        border-radius: 0.5rem;
        padding: 1rem;
        margin-top: 1rem;
    }

    .entities-title {
        color: var(--text-color);
        font-size: 0.875rem;
        margin-bottom: 0.75rem;
        font-weight: 500;
    }

    .entity-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .entity-chip {
        background-color: var(--highlight-color);
        color: var(--text-color);
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        border: 1px solid rgba(76, 175, 80, 0.4);
    }

    /* Override Streamlit's default container padding */
    .element-container {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_metrics(model_dir):
    """Load model metrics from JSON file"""
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
    """Create a bar plot comparing metrics across splits"""
    logger.info("Generating metrics comparison plot")
    try:
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
            yaxis_gridcolor='lightgray',
            height=500
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error generating metrics plot: {str(e)}")
        raise

def plot_confusion_matrix(conf_matrix, split_name):
    """Create a heatmap of the confusion matrix"""
    logger.info(f"Generating confusion matrix for {split_name} split")
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
        
        return fig
    except Exception as e:
        logger.error(f"Error generating confusion matrix: {str(e)}")
        raise

def display_tagged_text(result, show_pos_tags=False):
    """Display tagged text with correct entity summary box structure"""
    with st.expander("Analysis Result", expanded=True):
        # Format tokens and entities
        html_parts = []
        tokens = result['sentence'].split()
        predictions = [pred for _, pred in result['predictions']]
        pos_tags = result['pos_tags']
        
        for i, (token, is_entity, pos_tag) in enumerate(zip(tokens, predictions, pos_tags)):
            pos_tag_html = f'<span class="pos-tag">{pos_tag}</span>' if show_pos_tags else ''
            
            if is_entity:
                html_parts.append(
                    f'<span class="entity-highlight">{token}{pos_tag_html}</span>'
                )
            else:
                html_parts.append(f'{token}{pos_tag_html}')
            
            if i < len(tokens) - 1 and not tokens[i+1] in {',', '.', '!', '?', ':', ';'}:
                html_parts.append(' ')

        # Create the main content
        st.markdown('<div class="section-header">Identified Entities</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="entity-text">{"".join(html_parts)}</div>', unsafe_allow_html=True)
        
        # Add entity summary with exact structure
        entities = [token for token, pred in result['predictions'] if pred == 1]
        if entities:
            entity_chips = ''.join([
                f'<span class="entity-chip">{entity}</span>'
                for entity in entities
            ])
            
            entity_box_html = f"""
                                <div class="entities-box">
                                    <div class="entities-title">Entity Summary</div>
                                    <div class="entity-container">
                                        {entity_chips}
                                    </div>
                                </div>"""
            st.markdown(entity_box_html, unsafe_allow_html=True)

def process_text_input(input_text, predictor, as_batch=False):
    """Process input text and display results"""
    try:
        with st.spinner("Processing..."):
            if as_batch:
                sentences = [s.strip() for s in input_text.split('\n') if s.strip()]
                results = predictor.predict_batch(sentences)
                
                st.subheader("Results")
                for result in results:
                    display_tagged_text(result, True)
            else:
                results = predictor.predict_batch([input_text])
                
                st.subheader("Results")
                display_tagged_text(results[0], True)
            
            # Display legend
            st.markdown("""
            <div class="legend-box">
                <h4>Legend:</h4>
                <div style="margin: 0.5em 0;">
                    <span class="entity-highlight">Highlighted text</span> - Named Entity
                </div>
                <div style="margin: 0.5em 0;">
                    <span class="pos-tag">Small text above</span> - Part of Speech (POS) Tag
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        error_msg = f"Error processing text: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)

def display_test_results(results, show_pos_tags=False):
    """Display test results with enhanced visualization"""
    if not results:
        return
    
    for idx, result in enumerate(results, 1):
        with st.container():
            st.markdown(
                f'<div class="result-container">',
                unsafe_allow_html=True
            )
            st.markdown(f"**Example {idx}**")
            display_tagged_text(result, show_pos_tags)
            st.markdown('</div>', unsafe_allow_html=True)

def display_metrics_tab(metrics):
    """Display model metrics tab content"""
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
                    'F1 Score': f"{split_metrics['f1']:.3f}",
                    'Threshold': f"{split_metrics['threshold']:.3f}"
                }
                detailed_metrics.append(row)
            
            st.table(pd.DataFrame(detailed_metrics))
            
        except Exception as e:
            error_msg = f"Error displaying metrics: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
    else:
        st.warning("No metrics found. Train the model first to see performance metrics.")

def display_test_examples_tab(predictor, test_categories):
    """Display test examples tab content"""
    st.header("Test Examples")
    
    category = st.selectbox(
        "Select Example Category",
        options=list(test_categories.keys()),
        help="Choose a category of test examples to analyze"
    )
    
    # Show category description
    st.markdown(f"""
    <div class="category-description">
        <strong>{category}</strong><br>
        Number of examples: {len(test_categories[category])}
    </div>
    """, unsafe_allow_html=True)
    
    # Process button with options
    col1, col2 = st.columns(2)
    with col1:
        show_pos = st.checkbox("Show POS Tags", value=True)
    
    if st.button("Process Examples"):
        with st.spinner("Processing test examples..."):
            test_sentences = test_categories[category]
            results = predictor.predict_batch(test_sentences)
            
            if results:
                # Calculate category statistics
                total_entities = sum(len([token for token, pred in r['predictions'] if pred == 1]) 
                                   for r in results)
                avg_entities = total_entities / len(results)
                
                # Display statistics
                st.markdown(f"""
                <div class="statistics-box">
                    <strong>Category Statistics:</strong><br>
                    Total Entities: {total_entities}<br>
                    Average Entities per Sentence: {avg_entities:.2f}
                </div>
                """, unsafe_allow_html=True)
                
                # Display results
                display_test_results(results, show_pos)
                
                # Add download option
                results_df = pd.DataFrame([{
                    'Input': r['sentence'],
                    'Tagged Output': r['tagged_text'],
                    'POS Tags': ' | '.join([f"{t}({p})" for t, p in zip(r['sentence'].split(), r['pos_tags'])]),
                    'Entities': ', '.join([token for token, pred in r['predictions'] if pred == 1])
                } for r in results])
                
                st.download_button(
                    label="Download Results",
                    data=results_df.to_csv(index=False),
                    file_name=f"ner_results_{category.lower().replace(' ', '_')}.csv",
                    mime="text/csv"
                )

def main():
    """Main function for the Streamlit application"""
    logger.info("Starting Named Entity Recognition System")
    
    st.title("Named Entity Recognition System 🏷️")
    
    # Initialize predictor
    try:
        predictor = NERPredictor("model")
        model_loaded = True
        # Get test categories from predictor
        TEST_CATEGORIES, _ = predictor.get_test_cases()
        logger.info("Successfully initialized NER predictor")
    except Exception as e:
        error_msg = f"Error initializing model: {str(e)}"
        logger.error(error_msg)
        st.error(error_msg)
        model_loaded = False
        TEST_CATEGORIES = {}
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["NER Inference", "Model Metrics", "Test Examples"])
    
    # Tab 1: NER Inference
    with tab1:
        st.header("Named Entity Recognition")
        st.write("""
        Enter text to identify named entities. The system will highlight named entities 
        and show their POS (Part of Speech) tags.
        """)
        
        # Text input
        input_text = st.text_area(
            "Input Text",
            value="Washington DC is the capital of United States of America.",
            height=100,
            help="Enter the text you want to analyze. Multiple sentences are supported."
        )
        
        # Add batch processing option
        process_as_batch = st.checkbox(
            "Process as separate sentences",
            value=False,
            help="When checked, each line will be processed separately."
        )
        
        if st.button("Identify Named Entities") and model_loaded:
            process_text_input(input_text, predictor, process_as_batch)
    
    # Tab 2: Model Metrics
    with tab2:
        st.header("Model Performance Metrics")
        metrics = load_metrics("model")
        display_metrics_tab(metrics)
    
    # Tab 3: Test Examples
    with tab3:
        if model_loaded and TEST_CATEGORIES:
            display_test_examples_tab(predictor, TEST_CATEGORIES)
        else:
            st.error("Model not loaded or no test cases available.")

if __name__ == "__main__":
    main()
