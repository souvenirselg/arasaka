import torch
import time
import json
import os
from datetime import datetime
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model and tokenizer
MODEL_PATH = "fine_tuned_gpt2_psychologist"
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token  # Ensure proper padding

# PHQ-9 Depression Questionnaire
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things?",
    "Feeling down, depressed, or hopeless?",
    "Trouble falling or staying asleep, or sleeping too much?",
    "Feeling tired or having little energy?",
    "Poor appetite or overeating?",
    "Feeling bad about yourself or that you are a failure?",
    "Trouble concentrating on things?",
    "Moving or speaking slowly, or being restless?",
    "Thoughts that you would be better off dead or hurting yourself?"
]

# GAD-7 Anxiety Questionnaire
GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge?",
    "Not being able to stop or control worrying?",
    "Worrying too much about different things?",
    "Trouble relaxing?",
    "Being so restless that it is hard to sit still?",
    "Becoming easily annoyed or irritable?",
    "Feeling afraid as if something awful might happen?"
]

def ask_questions(questions):
    responses = []
    for q in questions:
        while True:
            try:
                ans = int(input(f"{q} (0-3): "))
                if ans in [0, 1, 2, 3]:
                    responses.append(ans)
                    break
                else:
                    print("Invalid input. Please enter a number between 0 and 3.")
            except ValueError:
                print("Invalid input. Please enter a number between 0 and 3.")
    return sum(responses)

def assess_mental_state():
    print("\n🧾 Please answer the following questions based on your experiences over the past two weeks.")
    print("Answer each question on a scale of 0 (Not at all) to 3 (Nearly every day).\n")

    phq9_score = ask_questions(PHQ9_QUESTIONS)
    gad7_score = ask_questions(GAD7_QUESTIONS)

    if phq9_score >= 20 or gad7_score >= 15:
        severity = "severe"
        advice = "🔴 Severe symptoms detected. Please consider seeking professional help."
    elif phq9_score >= 10 or gad7_score >= 8:
        severity = "moderate"
        advice = "🟠 Moderate symptoms detected. Consider talking to a professional or using self-help strategies."
    else:
        severity = "minimal"
        advice = "🟢 Minimal symptoms detected. Maintaining healthy habits is recommended."

    print("\nAssessment Results:")
    print(f"PHQ-9 Depression Score: {phq9_score}/27")
    print(f"GAD-7 Anxiety Score: {gad7_score}/21")
    print(advice)

    print("\n📚 Suggested Resources:")
    print("- https://www.mhanational.org/")
    print("- https://www.nimh.nih.gov/health/topics")
    print("- Crisis support: Text HOME to 741741 (US)")

    result = {
        "timestamp": str(datetime.now()),
        "phq9_score": phq9_score,
        "gad7_score": gad7_score,
        "severity": severity
    }

    # Optional logging to file
    save_results = input("\nWould you like to save your results for future reference? (y/n): ").strip().lower()
    if save_results == 'y':
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(result, f, indent=2)
        print("✅ Results saved to logs/")

    return result

def clean_input(text):
    return ' '.join(text.strip().split())

def chat():
    print("\n🧠 AI Therapist Chatbot (type 'exit' to quit)")
    chat_history = ""

    while True:
        user_input = input("\nYou: ")
        user_input = clean_input(user_input)
        if user_input.lower() == 'exit':
            print("👋 Take care! Goodbye!")
            break

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

        response = tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        chat_history += f" {response}\n"
        elapsed_time = time.time() - start_time

        print(f"Therapist: {response}")
        print(f"⏱️ Response generated in {elapsed_time:.2f} seconds")

def main():
    print("\n🌿 Welcome to the Mental Health Support Assistant 🌿")
    while True:
        print("\nWhat would you like to do?")
        print("1. Take a mental health screening (PHQ-9 + GAD-7)")
        print("2. Talk to the AI therapist")
        print("3. Exit")

        choice = input("Enter your choice (1/2/3): ").strip()
        if choice == '1':
            assess_mental_state()
            chat()
            break
        elif choice == '2':
            chat()
            break
        elif choice == '3':
            print("👋 Take care! Stay well.")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()