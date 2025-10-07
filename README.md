# Sentiment + Sarcasm Detection (with Irony & Multipolarity)

This project combines transformer-based fine-tuning and an interactive Gradio web interface to analyze text for sentiment, sarcasm, irony, and multipolarity.
It consists of two core fine-tuned models (single-label and multi-label) built on Hugging Face Transformers, integrated into a unified app for both text and CSV inputs.

## Features 

- Fine-tuned Sentiment Model (Positive, Neutral, Negative) using cardiffnlp/twitter-roberta-base-sentiment
- Multi-label Model detecting:
    - Sarcasm
    - Irony
    - Multipolarity
- Interactive Gradio App
    - Analyze individual texts or entire CSV files
    - Real-time progress updates for large datasets
    - Downloadable CSV results
- Robust text cleaning and validation pipeline
- Early stopping and best model saving
- Per-label threshold control for improved multi-label precision

## Project Structure 

📁 Sentiment-Analysis/ 
│ 
├── model_tuning.ipynb / model_tuning.py # Fine-tunes sentiment classifier 
├── multilabel_tuning.py                 # Fine-tunes sarcasm/irony/multipolarity model 
├── gradio_app.py                        # Interactive Gradio interface 
├── training_data.csv                    # Input dataset for sentiment model 
├── sarcasm_data.csv                     # Input dataset for multi-label model 
├── fine_tuned_sentiment_model/          # Saved model & tokenizer (after training) 
├── fine_tuned_multilabel_model/         # Saved model & tokenizer (after training) 
└── README.md

## Installation 

1. Clone the Repository
2. Create and activate virtual environment
   python -m venv .venv
   .venv\Scripts\activate      # on Windows
   source .venv/bin/activate   # on macOS/Linux
3. Install dependencies
   pip install -r requirements.txt

## Model Tuning
1. Sentiment model
  Trains a 3-class sentiment classifier on the training_data.csv.
  python model_tuning.py

2. Multi-label model
  Trains sarcasm, irony, and multipolarity detectors.
  python multilabel_tuning.py
  Both models save their weights to:
  fine_tuned_sentiment_model/
  fine_tuned_multilabel_model/

## Launch Gradio interface
After fine-tuning, start the app locally:
python gradio_app.py
Open the displayed localhost URL (usually http://127.0.0.1:7860) in your browser to:
  - Paste or type text input for instant results
  - Upload CSVs for batch analysis
  - Download predictions as a CSV

## Key libraries
  - transformers
  - datasets
  - evaluate
  - torch
  - gradio
  - pandas
  - numpy
  - scikit-learn
  
