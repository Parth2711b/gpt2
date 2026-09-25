# GPT-2 from scratch

GPT-2 (124M) inference written in C++20 with no ML framework, plus a Python fine-tuning pipeline using HuggingFace.

The C++ engine loads weights exported from HuggingFace, runs the full forward pass, and generates text using the original BPE tokenizer — all in a single file.

---

## What's inside

```
inference/      C++20 inference engine (main.cc)
modelLoader/    Python scripts: weight export (main.py) and SFT training (train_sft.py)
weights/        Base GPT-2 weights (text or binary)
weights_finetuned/  Weights after fine-tuning on Alpaca
```

---

## Build & run inference

```bash
cd inference
g++ -std=c++20 -O3 -march=native main.cc -o gpt2

# single prompt
./gpt2 "The meaning of life is" 60 0.8 40 0.9 1.2

# use fine-tuned weights
./gpt2 "### Instruction: explain gravity" 80 0.7 40 0.9 1.1 --weights ../weights_finetuned

# interactive mode (launched after the optional one-shot prompt)
./gpt2
```

Positional arguments: `prompt  tokens  temperature  topK  topP  repetitionPenalty`  
`temperature 0` → greedy decoding. All fields have sensible defaults.

In interactive mode every field is editable per prompt — just press Enter to keep the current value.

---

## Export weights from HuggingFace

```bash
cd modelLoader
pip install transformers torch

python main.py
```

Dumps all weight tensors to `../weights/` as space-separated float32 `.txt` files alongside `vocab.json` and `merges.txt`.

The C++ loader picks up a binary `.bin` file over a `.txt` file of the same name if present, which is much faster to load.

---

## Fine-tuning on Alpaca (SFT)

```bash
cd modelLoader
python train_sft.py
```

Trains for 1 epoch on 200 Alpaca examples (easy to adjust at the top of the file), then exports to `../weights_finetuned/`. Uses Apple MPS automatically when available.

The prompt format follows the Alpaca template so the instruction/response split is masked correctly during training.

---

## Inference features

- KV cache — prompt tokens are processed once, then only the new token runs through the model each step
- Sampling — top-k, top-p (nucleus), temperature, and HF-style repetition penalty
- Tied LM head — uses the embedding matrix directly for logit computation (no separate weight)
- Binary weight loading — place a `.bin` (raw float32, same byte order) next to any `.txt` to skip text parsing
- Streaming output — prints tokens as they are generated; handles multi-byte UTF-8 characters split across token boundaries correctly

---

## Model config

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Embedding dim | 768 |
| Attention heads | 12 |
| MLP dim | 3072 |
| Vocabulary | 50257 |
| Context length | 1024 |
