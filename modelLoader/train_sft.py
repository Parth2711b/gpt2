"""
Step 1 of train_sft.py: load the Alpaca dataset, build (input_ids, labels) pairs
with the prompt portion masked out (-100), and print one example to verify the
masking is correct before we add the actual training loop.
"""

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

MODEL_NAME = "GPT2"
NUM_EXAMPLES = 1000   # trained on a larger subset for better quality

# --- load base model + tokenizer (same starting point as main.py) ---
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token by default

model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# --- load a small subset of Alpaca ---
raw_dataset = load_dataset("tatsu-lab/alpaca", split=f"train[:{NUM_EXAMPLES}]")


def build_example(row):
    """
    Turns one Alpaca row into (input_ids, labels).
    labels has -100 over the prompt portion, real token ids over the response portion.
    """
    if row["input"]:
        prompt_text = (
            f"### Instruction:\n{row['instruction']}\n\n"
            f"### Input:\n{row['input']}\n\n"
            f"### Response:\n"
        )
    else:
        prompt_text = (
            f"### Instruction:\n{row['instruction']}\n\n"
            f"### Response:\n"
        )

    response_text = row["output"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    response_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + response_ids
    labels = [-100] * len(prompt_ids) + response_ids

    return {"input_ids": input_ids, "labels": labels}


# --- sanity check on one example, before touching the whole dataset ---
example = build_example(raw_dataset[0])

print("Instruction:", raw_dataset[0]["instruction"])
print("Output:", raw_dataset[0]["output"])
print()
print("Total tokens:", len(example["input_ids"]))
print("Masked (-100) tokens:", example["labels"].count(-100))
print()
print("First few labels (prompt, should be -100):", example["labels"][:5])
print("Last few labels (response, should be real ids):", example["labels"][-5:])
print("Decoded last few labels:", tokenizer.decode(example["labels"][-5:]))


# =========================================================================
# Step 2: build the full dataset, batch with padding, and train.
# =========================================================================

processed_dataset = [build_example(row) for row in raw_dataset]


def collate_batch(batch):
    """
    Pads a list of (input_ids, labels) examples to the same length so they
    can be stacked into one tensor. Padding positions get attention_mask=0
    (model ignores them) and labels=-100 (no loss on them, same as the
    prompt masking above).
    """
    max_length = max(len(item["input_ids"]) for item in batch)
    pad_id = tokenizer.pad_token_id

    input_ids_batch = []
    labels_batch = []
    attention_mask_batch = []

    for item in batch:
        pad_amount = max_length - len(item["input_ids"])
        input_ids_batch.append(item["input_ids"] + [pad_id] * pad_amount)
        labels_batch.append(item["labels"] + [-100] * pad_amount)
        attention_mask_batch.append([1] * len(item["input_ids"]) + [0] * pad_amount)

    return {
        "input_ids": torch.tensor(input_ids_batch),
        "labels": torch.tensor(labels_batch),
        "attention_mask": torch.tensor(attention_mask_batch),
    }


BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 5e-5

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nTraining on device: {device}")

model.to(device)
model.train()   # enables dropout etc. (training mode, vs eval mode for inference)

dataloader = DataLoader(processed_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# setup learning rate scheduler
total_steps = len(dataloader) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # forward pass: HF computes cross-entropy loss internally, using
        # ignore_index=-100 on the labels we built above
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()        # compute gradients for every weight

        # clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()       # nudge every weight against its gradient
        scheduler.step()       # update learning rate
        optimizer.zero_grad()  # clear gradients before the next batch

        if step % 20 == 0:
            print(f"epoch {epoch} step {step}/{len(dataloader)}  loss={loss.item():.4f}")

print("\nTraining done.")


# =========================================================================
# Step 3: export fine-tuned weights, same format as main.py
# =========================================================================

import os

OUTPUT_DIR = "../weights_finetuned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model.eval()
model.to("cpu")   # move back to cpu before exporting, simplest for numpy conversion

for name, param in model.named_parameters():
    values = param.detach().numpy().flatten()
    with open(f"{OUTPUT_DIR}/{name}.txt", "w") as f:
        f.write(" ".join(str(v) for v in values))

tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")
print(f"Exported fine-tuned weights to {OUTPUT_DIR}")