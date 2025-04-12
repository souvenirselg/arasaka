from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load model and tokenizer
MODEL_PATH = "fine_tuned_gpt2_psychologist"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

chat_history = ""

def clean_input(text):
    return ' '.join(text.strip().split())

# Chat function to generate model response
def generate_response(input_text):
    text = clean_input(input_text)
    global chat_history
    chat_history += f"You: {text}\nTherapist:"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    output = model.generate(
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.92,
    repetition_penalty=1.2
)

    response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
    response = response.strip().split("\n")[0]
    return response

# API route to handle chat messages
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    input_text = data.get('input')
    print("Input text:", input_text)
    emotion = data.get('emotion', 'Neutral')
    
    # Generate model's response
    model_response = generate_response(input_text)
    print("Model response:", model_response)
    
    return jsonify({"response": model_response})

# API route for emotion detection (simulated for now)
@app.route('/emotion', methods=['GET'])
def get_emotion():
    # This can be dynamically changed based on real-time emotion detection
    emotion = "Happy"  # Static for now
    return jsonify({"emotion": emotion})

if __name__ == '__main__':
    app.run(debug=True)
