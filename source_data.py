import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import gradio as gr

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Function for single review
def analyze_sentiment(text):
    result = classifier(text)[0]
    return f"{result['label']} ({result['score']:.2f})"

# Function for file upload
def analyze_file(file):
    df = pd.read_csv(file.name)  # assuming column "review"
    if 'review' not in df.columns:
        return "CSV must contain a 'review' column.", None

    results = classifier(list(df['review']))
    df['sentiment'] = [r['label'] for r in results]
    df['confidence'] = [r['score'] for r in results]
    output_path = "classified_reviews.csv"
    df.to_csv(output_path, index=False)
    return "File processed successfully.", output_path

# Build Gradio UI
interface = gr.Interface(
    fn=analyze_sentiment,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis Tool",
    description="Enter a single review for analysis below, or upload a file in the second tab."
)

file_interface = gr.Interface(
    fn=analyze_file,
    inputs=gr.File(file_types=[".csv"]),
    outputs=["text", gr.File(label="Download Result")],
    title="Batch Sentiment Classifier",
    description="Upload a CSV file with a column named 'review'."
)

gr.TabbedInterface([interface, file_interface], ["Single Review", "Batch File Upload"]).launch()