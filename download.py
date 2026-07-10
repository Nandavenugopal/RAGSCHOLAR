from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/flan-t5-base"
model_path = "saved_model/flan-t5-base"

# Download and save model and tokenizer locally
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
model.save_pretrained(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_path)