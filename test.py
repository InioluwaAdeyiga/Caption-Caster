import os
import re
from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import transformers
from langchain import HuggingFacePipeline
from langchain.prompts.prompt import PromptTemplate
from langchain import LLMChain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
import torch

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

model = "HuggingFaceH4/zephyr-7b-beta"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit =True,
    model_kwargs={ "temperature": 0.5, "max_length": 500, "min_length": 50,"top_p":0.9, "no_repeat_ngram_size":2, "do_sample":True},
)

llm = HuggingFacePipeline(pipeline=pipeline)

prompt = ChatPromptTemplate.from_messages([
    ("system","You are an AI chatbot that generates Instagram captions for marketing purposes, based on a prompt given by the user. The captions should be engaging, relevant to the target audience, and follow any specific guidelines provided in the query"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input_text}")

])

@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    input_text = request.json.get("text")
    if not input_text:
        return jsonify({"error": "No input text provided"}),

    chat_history = []
    chain = LLMChain(llm=llm, prompt=prompt)
    inputs = {
        "input_text": input_text,
        "chat_history" : chat_history
    }
    caption = chain.run(inputs)
    chat_history.append(HumanMessage(content=input_text))
    chat_history.append(AIMessage(content=caption))
    return jsonify({"caption": caption})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))