import os
import sys
from transformers import GPT2LMHeadModel, GPT2Tokenizer

OUTPUT_DIR = "../weights"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Loading GPT-2 from HuggingFace...")
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

params = list(model.named_parameters())
total = len(params)

print(f"Exporting {total} weight tensors to {OUTPUT_DIR}/")
for i, (name, param) in enumerate(params):
    values = param.detach().numpy().flatten()
    path = f"{OUTPUT_DIR}/{name}.txt"
    with open(path, "w") as f:
        f.write(" ".join(str(v) for v in values))
    if (i + 1) % 10 == 0 or (i + 1) == total:
        print(f"  [{i+1}/{total}] {name}")

tokenizer.save_pretrained(f"{OUTPUT_DIR}/tokenizer")
print(f"\nDone. Weights written to {OUTPUT_DIR}/")
print("Vocab and tokenizer saved to tokenizer/")