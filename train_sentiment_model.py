import time
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_scheduler
from torch.optim import AdamW
from datetime import datetime
from tqdm import tqdm

# 🚀 Load preprocessed data
print("\n📂 Loading preprocessed data...")
train_inputs, val_inputs, train_masks, val_masks = torch.load("model/preprocessed_data.pt")

# ⚠ Fix shape mismatches
assert train_inputs.shape == train_masks.shape, "Mismatch in training data shapes!"
assert val_inputs.shape == val_masks.shape, "Mismatch in validation data shapes!"
print(f"✅ Loaded training data with shape: {train_inputs.shape}")

# ⚡ Load tokenizer & model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# ✅ Set padding token (important for batch processing)
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# 📌 Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"💻 Training on: {device}")
model.to(device)

# 📦 Create DataLoader
batch_size = 4  # Reduce to avoid GPU memory issues
train_data = TensorDataset(train_inputs, train_masks)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

# ⚙ Define optimizer & scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
epochs = 3
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * epochs
)

# 🏋 Training loop
print("\n🚀 Starting training...\n")
model.train()

start_time = time.time()  # ⏱ Start timing
for epoch in range(epochs):
    epoch_start = time.time()  # Track epoch time
    total_loss = 0

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"🌀 Epoch {epoch+1}/{epochs}", unit="batch")):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 🚀 Prevent exploding gradients

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # ⏱ Show time every 10 batches
        if (step + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"⏱ [{datetime.now().strftime('%H:%M:%S')}] Step {step+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

    epoch_time = time.time() - epoch_start
    avg_loss = total_loss / len(train_dataloader)
    print(f"\n✅ Epoch {epoch+1} complete in {epoch_time:.2f} sec. Avg Loss: {avg_loss:.4f}\n")

# ⏳ Total training time
total_time = time.time() - start_time
print(f"\n🎉 Training finished in {total_time:.2f} seconds ({total_time/60:.2f} min)")

# 💾 Save model
print("\n💾 Saving fine-tuned model...")
model.save_pretrained("fine_tuned_gpt2_psychologist")
tokenizer.save_pretrained("fine_tuned_gpt2_psychologist")
print("✅ Model saved successfully!")
