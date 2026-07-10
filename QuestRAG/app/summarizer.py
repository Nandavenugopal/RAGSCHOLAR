from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import os

def summarize_pdf(pages):
    full_text = " ".join(page.page_content for page in pages)

    model_path = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    # Summarize using direct model inference
    prompt = f"summarize: {full_text[:2000]}"
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary