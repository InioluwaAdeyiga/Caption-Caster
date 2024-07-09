import os
import re
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from functools import lru_cache

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@lru_cache(maxsize=1)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("google/flan-t5-small")  # Use the smallest model

@lru_cache(maxsize=1)
def get_model():
    return AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")  # Use the smallest model

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    input_text = request.json.get("text")
    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    tokenizer = get_tokenizer()
    model = get_model()

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    # Environment variables for configuration
    num_return_sequences = int(os.getenv("NUM_RETURN_SEQUENCES", 1))
    max_length = int(os.getenv("MAX_LENGTH", 100))  # Reduce max length to limit memory usage
    min_length = int(os.getenv("MIN_LENGTH", 20))
    temperature = float(os.getenv("TEMPERATURE", 0.7))
    top_p = float(os.getenv("TOP_P", 0.9))
    repetition_penalty = float(os.getenv("REPETITION_PENALTY", 1.2))
    no_repeat_ngram_size = int(os.getenv("NO_REPEAT_NGRAM_SIZE", 2))

    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Use mixed precision
            output = model.generate(
                input_ids,
                max_length=max_length,
                min_length=min_length,
                attention_mask=attention_mask,
                num_return_sequences=num_return_sequences,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id
            )

    caption = tokenizer.decode(output[0], skip_special_tokens=True)

    # Efficient post-processing
    caption = re.sub(r"http\S+|www\S+|@\S+|#", "", caption)
    caption = re.sub(r"[^a-zA-Z0-9\s.,]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()

    return jsonify({"caption": caption})

if __name__ == "__main__":
    # Limit the memory usage for the model (in MB)
    torch.cuda.set_per_process_memory_fraction(0.5)  # Assume 520MB is around 0.5 of your GPU memory
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
