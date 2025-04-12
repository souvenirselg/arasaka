import cv2
from deepface import DeepFace
import threading
import torch
import time
import json
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "fine_tuned_gpt2_psychologist"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

# Shared variables
current_emotion = "Neutral"
running = True

# Facial emotion detection loop
def emotion_detector():
    global current_emotion, running
    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            current_emotion = result[0]['dominant_emotion']
        except:
            current_emotion = "Unknown"

        cv2.putText(frame, f"Emotion: {current_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Facial Emotion", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# Therapist model (placeholder)
def therapist_model_response(user_input, emotion):
    return f"(You look {emotion}) You said: {user_input}"

# Start the webcam/emotion detection thread
emotion_thread = threading.Thread(target=emotion_detector)
emotion_thread.start()

def clean_input(text):
    return ' '.join(text.strip().split())

def clean_response(text, max_sentences=2):
    sentences = text.strip().split('. ')
    return '. '.join(sentences[:max_sentences]) + ('.' if len(sentences) else '')

def chat(user_input, emotion):
    print("\n🧠 AI Therapist Chatbot (type 'exit' to quit)")
    chat_history = ""

    chat_history += f"You: {user_input}\nTherapist:"
    inputs = tokenizer(chat_history, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    start_time = time.time()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,  # <-- FIX: Enables temperature + top_p to actually work
        max_new_tokens=150,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    resp = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    response = clean_response(resp)
    chat_history += f" {resp}\n"
    elapsed_time = time.time() - start_time

    print(f"You look: {emotion} Therapist: {response}")
    print(f"⏱️ Response generated in {elapsed_time:.2f} seconds")


# Text input loop
print("Talk to your AI therapist (type 'exit' to quit):")
while running:
    try:
        text = input("You: ")
        user_input = clean_input(text)
        if user_input.lower() in ['exit', 'quit']:
            running = False
            break
        chat(user_input, current_emotion)
    except KeyboardInterrupt:
        running = False
        break

emotion_thread.join()
print("Session ended.")





