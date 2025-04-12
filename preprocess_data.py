import json
import torch
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split

data = []
with open("combined_dataset.json", "r", encoding="utf-8") as file:
    for line in file:
        try:
            data.append(json.loads(line.strip()))  # Read each line as a JSON object
        except json.JSONDecodeError as e:
            print(f"Skipping corrupted line: {e}")

print(f"✅ Successfully loaded {len(data)} entries!")

# ✅ Format data for training
formatted_data = []
for entry in data:
    user_text = entry.get('Context', '')  # Handle missing keys safely
    therapist_response = entry.get('Response', '')
    dialogue = f"User: {user_text}\nTherapist: {therapist_response}"
    formatted_data.append(dialogue)

# ✅ Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# ✅ Tokenize data
tokenized_data = tokenizer(
    formatted_data,
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Extract inputs
input_ids = tokenized_data["input_ids"]
attention_mask = tokenized_data["attention_mask"]

# 🔍 Debug: Check tokenized shapes
print(f"Tokenized input_ids shape: {input_ids.shape}")
print(f"Tokenized attention_mask shape: {attention_mask.shape}")

# ✅ Convert tensors to lists for consistent train-test split
input_ids_list = input_ids.tolist()
attention_mask_list = attention_mask.tolist()

# ✅ Train-validation split
train_input_ids, val_input_ids, train_attention_masks, val_attention_masks = train_test_split(
    input_ids_list, attention_mask_list, test_size=0.1, random_state=42
)

# ✅ Convert back to tensors
train_inputs = torch.tensor(train_input_ids)
val_inputs = torch.tensor(val_input_ids)
train_masks = torch.tensor(train_attention_masks)
val_masks = torch.tensor(val_attention_masks)

# 🔍 Debug: Check final shapes
print(f"Train Inputs shape: {train_inputs.shape}")
print(f"Train Masks shape: {train_masks.shape}")
print(f"Validation Inputs shape: {val_inputs.shape}")
print(f"Validation Masks shape: {val_masks.shape}")

# ✅ Shape validation
assert train_inputs.shape == train_masks.shape, f"❌ Mismatch: {train_inputs.shape} vs {train_masks.shape}"
assert val_inputs.shape == val_masks.shape, f"❌ Mismatch: {val_inputs.shape} vs {val_masks.shape}"

# ✅ Clone to avoid computation graph issues
train_inputs, val_inputs = train_inputs.clone().detach(), val_inputs.clone().detach()
train_masks, val_masks = train_masks.clone().detach(), val_masks.clone().detach()

# ✅ Save preprocessed tensors
torch.save((train_inputs, val_inputs, train_masks, val_masks), "preprocessed_data.pt")

print("✅ Preprocessing complete! Data saved successfully.")
