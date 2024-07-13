import os
import re
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import transformers
from langchain import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
import torch

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


model = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
    model_kwargs={"temperature": 0.5, "max_length": 300, "min_length": 50, "top_p": 0.9, "no_repeat_ngram_size": 2,
                  "do_sample": True},
)

llm = HuggingFacePipeline(pipeline=pipeline)
template = """
You are an AI chatbot that generates Instagram captions for marketing purposes, based on a prompt given by the user. The captions should be engaging, relevant to the target audience, and follow any specific guidelines provided in the query.

{query}
"""

prompt = PromptTemplate.from_messages([
    ("system",
     "You are an AI chatbot that generates Instagram captions for marketing purposes, based on a prompt given by the user. The captions should be engaging, relevant to the target audience, and follow any specific guidelines provided in the query"),

    ("human", "{query}")

])


@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    input_text = request.json.get("text")
    if not input_text:
        return jsonify({"error": "No input text provided"}),

    chat_history = []
    chain = LLMChain(llm=llm, prompt=prompt)
    chain_response = chain.run(input_text, chat_history)
    caption = chain_response

    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))