import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

MAX_LENGTH = 152
THRESHOLD = 0.5  # multi-label cutoff

# -----------------------
# Load Sentiment Model
# -----------------------
sentiment_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_sentiment_model")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_sentiment_model")
sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=sentiment_model,
    tokenizer=sentiment_tokenizer,
    truncation=True,
    max_length=MAX_LENGTH,
)

# -----------------------
# Load Sarcasm (multi-label) Model
# -----------------------
sarcasm_tokenizer = AutoTokenizer.from_pretrained("fine_tuned_multilabel_model")
sarcasm_model = AutoModelForSequenceClassification.from_pretrained("fine_tuned_multilabel_model")
sarcasm_pipe = pipeline(
    "text-classification",
    model=sarcasm_model,
    tokenizer=sarcasm_tokenizer,
    truncation=True,
    max_length=MAX_LENGTH,
)

# -----------------------
# Helpers
# -----------------------
sentiment_map_idx = {0: "Negative", 1: "Neutral", 2: "Positive"}

def nice_sentiment_label(raw_label: str) -> str:
    # Accept "POSITIVE"/"NEGATIVE"/"NEUTRAL" or "LABEL_x"
    lab = raw_label.strip()
    up = lab.upper()
    if up in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        return up.title()
    if lab.startswith("LABEL_") and lab.split("_")[-1].isdigit():
        idx = int(lab.split("_")[-1])
        return sentiment_map_idx.get(idx, lab)
    return lab  # fallback to whatever the model returns

def decode_multi(scores_list, threshold=THRESHOLD):
    """
    scores_list is the result for ONE text when calling
    sarcasm_pipe(..., return_all_scores=True). It looks like:
    [{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.8}, {'label': 'LABEL_2', 'score': 0.2}]
    Order corresponds to label indices (0,1,2).
    """
    # Ensure we have 3 scores in order 0,1,2
    idx_by_label = {}
    for item in scores_list:
        lab = item["label"]
        # Parse LABEL_n -> n
        if lab.startswith("LABEL_") and lab.split("_")[-1].isdigit():
            idx = int(lab.split("_")[-1])
        else:
            # fallback: try to infer index from position
            idx = None
        if idx is not None:
            idx_by_label[idx] = item["score"]

    s0 = idx_by_label.get(0, 0.0)  # Sarcasm
    s1 = idx_by_label.get(1, 0.0)  # Irony
    s2 = idx_by_label.get(2, 0.0)  # Multipolarity
    return (s0 >= threshold, s1 >= threshold, s2 >= threshold)

def yesno(b: bool) -> str:
    return "Yes" if b else "No"

# -----------------------
# Core analysis
# -----------------------
def analyze_texts_markdown(texts):
    if isinstance(texts, str):
        texts = [texts]
    # clean empty lines
    texts = [t.strip() for t in texts if str(t).strip()]
    if not texts:
        return "Provide at least one line of text."

    sar_preds = sarcasm_pipe(texts, batch_size=16, return_all_scores=True)
    sen_preds = sentiment_pipe(texts, batch_size=16)

    # Build markdown exactly like your example
    out_lines = []
    for txt, sars, sen in zip(texts, sar_preds, sen_preds):
        sarcastic, irony, multipolarity = decode_multi(sars, threshold=THRESHOLD)

        sen_label = nice_sentiment_label(sen["label"])
        sen_score = sen["score"]

        out_lines.append(
            f"Sentiment: {sen_label} {sen_score:.2f}\n"
            f"Sarcasm: {yesno(sarcastic)}\n"
            f"Irony: {yesno(irony)}\n"
            f"Multipolarity: {yesno(multipolarity)}"
        )
        out_lines.append("---")  # separator between samples

    # drop trailing separator
    if out_lines and out_lines[-1] == "---":
        out_lines.pop()

    return "\n".join(out_lines)

def analyze_texts_df(texts):
    # For CSV tab: return a DataFrame with Yes/No strings and the original text
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.strip() for t in texts if str(t).strip()]
    if not texts:
        return pd.DataFrame(columns=["Sentiment", "Sarcasm", "Irony", "Multipolarity", "Review"])

    sar_preds = sarcasm_pipe(texts, batch_size=16, return_all_scores=True)
    sen_preds = sentiment_pipe(texts, batch_size=16)

    rows = []
    for txt, sars, sen in zip(texts, sar_preds, sen_preds):
        sarcastic, irony, multipolarity = decode_multi(sars, threshold=THRESHOLD)
        sen_label = nice_sentiment_label(sen["label"])
        sen_score = sen["score"]
        rows.append({
            "Sentiment": f"{sen_label} {sen_score:.3f}",
            "Sarcasm": yesno(sarcastic),
            "Irony": yesno(irony),
            "Multipolarity": yesno(multipolarity),
            "Review": txt
        })
    return pd.DataFrame(rows)


def analyze_from_csv(file):
    df = pd.read_csv(file.name)

    # Accept "review" or "reviews"
    colname = None
    for c in df.columns:
        if c.lower() in {"review", "reviews"}:
            colname = c
            break

    if colname is None:
        return pd.DataFrame([{
            "Sentiment": "CSV must have a 'review' or 'reviews' column",
            "Sarcasm": None,
            "Irony": None,
            "Multipolarity": None,
            "Review": None
        }])

    texts = df[colname].astype(str).tolist()
    return analyze_texts_df(texts)

def analyze_from_csv_with_progress(file, total_rows, progress_callback=None):
    """Analyze CSV with progress tracking"""
    df = pd.read_csv(file.name)

    # Accept "review" or "reviews"
    colname = None
    for c in df.columns:
        if c.lower() in {"review", "reviews"}:
            colname = c
            break

    if colname is None:
        return pd.DataFrame([{
            "Sentiment": "CSV must have a 'review' or 'reviews' column",
            "Sarcasm": None,
            "Irony": None,
            "Multipolarity": None,
            "Review": None
        }])

    texts = df[colname].astype(str).tolist()
    
    # Process in smaller batches to show progress
    batch_size = 16
    all_results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_results = analyze_texts_df(batch_texts)
        all_results.append(batch_results)
        
        # Update progress with callback if provided
        processed = min(i + batch_size, len(texts))
        if progress_callback:
            progress_callback(processed, len(texts))
    
    # Combine all results
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
    else:
        final_df = pd.DataFrame(columns=["Sentiment", "Sarcasm", "Irony", "Multipolarity", "Review"])
    
    return final_df


def on_analyze_click(multiline_text):
    texts = multiline_text.split("\n")
    return analyze_texts_markdown(texts)

# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks() as interface:
    gr.Markdown("## Sentiment + Sarcasm (with Irony & Multipolarity) Detection")

    with gr.Tab("Text Input"):
        input_texts = gr.Textbox(
            lines=6,
            placeholder="Enter one text per line",
            label="Input"
        )
        output_md = gr.Markdown()
        run_btn = gr.Button("Analyze")
        run_btn.click(fn=on_analyze_click, inputs=input_texts, outputs=output_md)

    with gr.Tab("CSV Upload"):
          file_input = gr.File(label="Upload CSV with a 'review' or 'reviews' column", file_types=[".csv"])
          
          analyze_btn = gr.Button("Analyze CSV", visible=False)
          progress_text = gr.Textbox(label="Progress", value="", visible=False, interactive=False)
          
          file_output = gr.DataFrame(
                    headers=["Review","Sentiment", "Sarcasm", "Irony", "Multipolarity"],
                    datatype=["str", "str", "str", "str", "str"],
                    row_count=(1, "dynamic"),
                    visible=False
          )

          download_btn = gr.DownloadButton(
                    label="Download Results as CSV",
                    visible=False
          )

          def on_file_upload(file):
                    if file is None:
                              return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                    
                    # Show analyze button when file is uploaded
                    return (
                              gr.update(visible=True),  # analyze button
                              gr.update(visible=False), # progress text
                              gr.update(visible=False), # file output
                              gr.update(visible=False)  # download button
                    )

          def process_csv_with_progress(file):
                    if file is None:
                              return (
                                        gr.update(value="No file uploaded", visible=True),
                                        gr.update(visible=False),
                                        gr.update(visible=False)
                              )
                    
                    # Read CSV to get row count for progress tracking
                    try:
                              df = pd.read_csv(file.name)
                              total_rows = len(df)
                              
                              # Show initial progress
                              yield (
                                        gr.update(value=f"Processing 0/{total_rows} rows...", visible=True),
                                        gr.update(visible=False),
                                        gr.update(visible=False)
                              )
                              
                              # Accept "review" or "reviews"
                              colname = None
                              for c in df.columns:
                                        if c.lower() in {"review", "reviews"}:
                                                  colname = c
                                                  break

                              if colname is None:
                                        error_df = pd.DataFrame([{
                                                  "Sentiment": "CSV must have a 'review' or 'reviews' column",
                                                  "Sarcasm": None,
                                                  "Irony": None,
                                                  "Multipolarity": None,
                                                  "Review": None
                                        }])
                                        yield (
                                                  gr.update(value="Error: CSV must have a 'review' or 'reviews' column", visible=True),
                                                  gr.update(value=error_df, visible=True),
                                                  gr.update(visible=False)
                                        )
                                        return

                              texts = df[colname].astype(str).tolist()
                              
                              # Process in smaller batches with progress updates
                              batch_size = 16
                              all_results = []
                              
                              for i in range(0, len(texts), batch_size):
                                        batch_texts = texts[i:i+batch_size]
                                        batch_results = analyze_texts_df(batch_texts)
                                        all_results.append(batch_results)
                                        
                                        # Update progress after each batch
                                        processed = min(i + batch_size, len(texts))
                                        yield (
                                                  gr.update(value=f"Processing {processed}/{total_rows} rows...", visible=True),
                                                  gr.update(visible=False),
                                                  gr.update(visible=False)
                                        )
                              
                              # Combine all results
                              if all_results:
                                        results_df = pd.concat(all_results, ignore_index=True)
                              else:
                                        results_df = pd.DataFrame(columns=["Sentiment", "Sarcasm", "Irony", "Multipolarity", "Review"])
                              
                              # Save to temporary CSV file for download
                              import tempfile
                              temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
                              results_df.to_csv(temp_file.name, index=False)
                              temp_file.close()
                              
                              # Show completion and results
                              yield (
                                        gr.update(value=f"Completed! Processed {len(results_df)} rows.", visible=True),
                                        gr.update(value=results_df, visible=True),
                                        gr.update(visible=True, value=temp_file.name)
                              )
                              
                    except Exception as e:
                              yield (
                                        gr.update(value=f"Error processing file: {str(e)}", visible=True),
                                        gr.update(visible=False),
                                        gr.update(visible=False)
                              )

          file_input.upload(
                    fn=on_file_upload, 
                    inputs=file_input, 
                    outputs=[analyze_btn, progress_text, file_output, download_btn]
          )
          
          analyze_btn.click(
                    fn=process_csv_with_progress,
                    inputs=file_input,
                    outputs=[progress_text, file_output, download_btn]
          )


interface.launch()
