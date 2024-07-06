import os
import re
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

def load_model_and_tokenizer():
    base_path = os.path.dirname(__file__)
    relative_path = "final_parm"
    full_path = os.path.join(base_path, relative_path)
    tokenizer = AutoTokenizer.from_pretrained(full_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(full_path)
    return tokenizer, model

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    input_text = request.json.get("text")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    tokenizer, model = load_model_and_tokenizer()

    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    num_return_sequences = 1
    max_length = 300
    min_length = 100
    temperature = 0.5
    top_p = 0.9
    repetition_penalty = 1.5
    no_repeat_ngram_size = 2

    with torch.no_grad():
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

    # Post-processing to remove links, social media handles, and special characters
    caption = re.sub(r"http\S+|www\S+|@\S+|#", "", caption)
    caption = re.sub(r"[^a-zA-Z0-9\s.,]", "", caption)
    caption = re.sub(r"\s+", " ", caption).strip()

    return jsonify({"caption": caption})

if __name__ == "__main__":
    app.run(debug=True)
