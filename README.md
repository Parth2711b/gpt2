# GPT-2 from scratch

GPT-2 (124M) inference written in C++20 with no ML framework, plus a Python fine-tuning pipeline using HuggingFace.

The C++ engine loads weights exported from HuggingFace, runs the full forward pass, and generates text using the original BPE tokenizer — all in a single file.

---

## What's inside

```
inference/          C++20 inference engine (main.cc)
modelLoader/        Python scripts: weight export (main.py) and SFT training (train_sft.py)
weights/            Base GPT-2 weights (text or binary)
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

The C++ loader picks up a `.bin` file (raw float32, same byte order) over a `.txt` of the same name when present — much faster to load.

---

## Fine-tuning on Alpaca (SFT)

```bash
cd modelLoader
python train_sft.py
```

Trains for 1 epoch on 200 Alpaca examples (configurable at the top of the file), then exports to `../weights_finetuned/`. Uses Apple MPS automatically when available.

---

## Inference features

- **KV cache** — prompt tokens processed once; only the new token runs per step
- **Sampling** — top-k, top-p (nucleus), temperature, repetition penalty
- **Tied LM head** — embedding matrix reused for logit projection
- **Binary weights** — `.bin` fast path over `.txt` parsing
- **Streaming output** — handles multi-byte UTF-8 split across token boundaries

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
