# GPT-2 from Scratch

GPT-2 (124M) inference written in C++20 with no ML framework, plus a Python fine-tuning
pipeline using HuggingFace. The C++ engine loads weights exported from HuggingFace, runs
the full forward pass by hand, and generates text using the original BPE tokenizer — all
in a single file, no PyTorch/TensorFlow dependency at inference time.

---

## What's inside

```
inference/          C++20 inference engine (main.cc)
modelLoader/         Python scripts: weight export (main.py), SFT training (train_sft.py), binary converter (convert_to_bin.py)
weights/             Base GPT-2 weights (text or binary)
weights_finetuned/   Weights after fine-tuning on Alpaca
```

---

## How a prompt becomes text — the full forward pass

This is what happens, in order, every time the model processes one token. It's the same
sequence for every token in the prompt and every token generated afterward — the only
difference is what gets cached and skipped (see KV cache, below).

### 1. Tokenization (byte-level BPE)

The raw prompt string is first split into rough chunks by `preTokenize` — a hand-written
version of GPT-2's regex (words, numbers, punctuation runs, contractions like `'re`/`'ll`
handled as their own chunks), since `std::regex` has no Unicode-category support.

Each chunk is then mapped byte-by-byte into GPT-2's "visible" byte-to-unicode alphabet
(`byteToUnicode`, built once by `buildByteTables`) and repeatedly merged using the ranked
merge rules from `merges.txt` (`bpeEncode`): at each step, find the highest-priority
adjacent pair present in `mergeRanks` and fuse it into one symbol, until no known merge
applies. Every final symbol is looked up in `vocab.json` (`tokenToId`) to get an integer
token id. `encode()` just runs this over every chunk and concatenates the ids.

Result: a `std::vector<int>` of token ids — this is the model's actual input, text never
touches the numeric code again until decoding at the very end.

### 2. Token embedding + positional embedding

Each token id indexes into `wte` (`embeddingWeights`, shape `[50257][768]`) to get a
768-dimensional vector representing "what token this is." Each *position* in the sequence
(0, 1, 2, …) indexes into `wpe` (`positionalEmbeddingWeights`, shape `[1024][768]`) to get
a vector representing "where in the sequence this token is." These two vectors are added
elementwise — this sum is `x`, the model's working representation of that token, and it's
what enters the first transformer block (`gptStep`).

### 3. Transformer block (repeated 12 times — `transformerBlockStep`)

Each of the 12 layers does the same four-stage computation, using a pre-norm residual
design (normalize, transform, add back to the input — this is what keeps deep networks
like this trainable):

**a. LayerNorm 1** (`layerNorm`) — normalizes `x` to zero mean / unit variance per
position, then applies a learned per-channel scale and shift (`lnAttWeights`/`lnAttBiases`).
This stabilizes the scale of activations going into attention.

**b. Multi-head causal self-attention**
- The normalized vector is projected three separate ways — **Q**uery, **K**ey, **V**alue
  (`forwardPass` with `qWeights`/`kWeights`/`vWeights`) — each still 768-dim. These three
  projections are what "QKV" refers to: three different linear views of the same input,
  used to ask "what am I looking for" (Q), "what do I contain" (K), and "what do I offer
  if attended to" (V).
- The 768 dimensions are split into 12 heads of 64 dims each (`kHeadDim`), so attention
  runs independently in 12 parallel subspaces — this lets different heads specialize in
  different kinds of relationships (e.g. one head tracking nearby words, another tracking
  long-range structure).
- For each head, the new token's Q is dot-producted against every *previous* token's K
  (including itself) that's sitting in the `KvCache` — this dot product measures
  similarity/relevance. The causal part is implicit: the cache only ever contains this
  token and earlier ones, so there is structurally no way to attend to future tokens.
- Scores are divided by `sqrt(64)` (`scale`) to stop the dot products from growing large
  and flattening the softmax's gradient as head-dimension increases (standard scaled
  dot-product attention).
- **Softmax** (`softmax`) turns these raw scores into a probability distribution over all
  attended positions — this is what "attention weights" actually are: how much of each
  past token's information gets pulled into the current position.
- Each past token's V vector is then weighted by its attention probability and summed —
  this weighted sum, across all 12 heads concatenated back into 768 dims, is what actually
  flows forward as `attended`.
- A final linear projection (`oWeights`/`oBiases`) mixes the concatenated heads back
  together, and the result is added back onto the original `x` (residual connection).

**c. LayerNorm 2** — same normalization idea, applied before the MLP (`lnMlpWeights`/
`lnMlpBiases`).

**d. MLP / feed-forward block**
- `c_fc`: a linear expansion from 768 → 3072 dims (`l1Weights`/`l1Biases`).
- **GELU** (`gelu`) — a smooth, differentiable nonlinearity applied elementwise; GPT-2
  specifically uses the tanh approximation of GELU (`gelu_new`). This is what gives the
  network its actual expressive nonlinearity — without it, stacking linear layers would
  collapse into one linear layer no matter how deep the network is.
- `c_proj`: a linear projection back down from 3072 → 768 dims (`l2Weights`/`l2Biases`).
- The result is added back onto the residual stream from step (b).

That combined residual output becomes the `x` fed into the next layer. After 12 of these,
`x` has been repeatedly refined by attention (mixing information across positions) and MLP
(nonlinear per-position transformation).

### 4. Final LayerNorm + tied LM head

After the 12th layer, one more LayerNorm (`finalNormWeights`/`finalNormBiases`) is applied.
The result is then compared against *every* row of the same `wte` embedding matrix used in
step 2 via a dot product — this produces one raw score (a "logit") per vocabulary entry,
50257 numbers total. Reusing `wte` for this instead of a separate learned output matrix is
what "tied embeddings" means — it halves the parameters spent on the embedding/output
layer and is standard in GPT-2.

### 5. Sampling — turning logits into the next token

- **Greedy** (`argmaxToken`) — just take the highest-scoring token. Deterministic, often
  repetitive.
- **Temperature** — logits are divided by `temperature` before softmax; below 1.0 sharpens
  the distribution (more confident/conservative), above 1.0 flattens it (more random).
- **Top-k** — only the `k` highest-logit tokens are kept as candidates at all; the rest are
  discarded before sampling (`sampleToken`).
- **Top-p / nucleus** — on top of the top-k candidates (already sorted by
  `std::partial_sort`), keep only the smallest prefix whose cumulative probability crosses
  `topP`, then renormalize just that subset. This adapts the candidate pool to how
  confident the model actually is at each step, instead of a fixed count.
- **Repetition penalty** (`applyRepetitionPenalty`) — before any of the above, logits for
  every token id already used earlier in the sequence are pulled toward zero (divided by
  the penalty if positive, multiplied if negative), discouraging the model from looping on
  the same tokens.

### 6. KV cache — why generation doesn't redo all the work every step

Without caching, generating token N would mean re-running the entire forward pass on all N
previous tokens from scratch. Instead, `KvCache` stores every layer's K and V vectors for
every token as they're computed. Generating a new token only runs the full forward pass
*once*, for that one new token — its Q is compared against the *already-computed* K/V of
every earlier token pulled from the cache. This is what makes step-by-step generation
roughly linear in total work rather than quadratic-and-repeated.

### 7. Decoding back to text

The chosen token id is looked up in `gpt2Tokens` and converted from GPT-2's internal
byte-to-unicode representation back to raw bytes (`decodeToken`). Because BPE token
boundaries don't respect UTF-8 character boundaries, a single multi-byte character (like a
curly quote) can be split across two separate tokens — printing each token's bytes the
instant it's produced can therefore emit a broken/incomplete UTF-8 sequence mid-character.
`incompleteUtf8TailLength` detects when the tail of the not-yet-printed buffer is an
incomplete multi-byte sequence and holds it back until the completing bytes arrive from the
next token, so the terminal only ever receives valid, complete UTF-8.

---

## What I built on top of the base inference loop

The forward pass above — embedding, attention, MLP, tied head — is the unmodified GPT-2
computation. Everything below is engine/tooling I added around it:

- **Single weight load, multi-prompt interactive loop** — weights are loaded once at
  startup and reused across an unlimited number of prompts in the same process, instead of
  reloading per run.
- **Per-round sampling controls** — `tokens`, `topP`, and `repetitionPenalty` are prompted
  for on every round (blank input keeps the previous default), so sampling behavior can be
  tuned without recompiling or restarting.
- **Repetition penalty** — not part of the original minimal sampler; added specifically
  because the base model tends to loop on phrases without it.
- **Top-p (nucleus) sampling** — layered on top of the original top-k-only sampler, to
  adapt the candidate pool size to the model's actual confidence at each step.
- **UTF-8-safe streaming** — fixed a real bug where printing tokens as they streamed could
  split a multi-byte character across two flushes and corrupt the terminal output.
- **Performance metrics** — prints the total tokens generated and the generation speed
  (`tok/s`) upon completion of each generation.
- **`--help` flag** — added built-in usage instructions.
- **`--weights <path>` flag** — lets the same binary load either `../weights` (base) or
  `../weights_finetuned` (after SFT) without touching source or recompiling, so base and
  fine-tuned behavior can be compared directly.

---

## Fine-tuning pipeline (SFT) — `train_sft.py`

`main.py` only ever loaded the base model and dumped its weights — no training happened
anywhere in the original code. `train_sft.py` adds actual supervised fine-tuning, using
the same weight-export logic as `main.py` at the end:

1. **Dataset** — a configurable subset of the Alpaca instruction dataset
   (`tatsu-lab/alpaca`) is loaded via HuggingFace `datasets`.
2. **Prompt template** — each example is formatted as:
   ```
   ### Instruction:
   {instruction}

   ### Response:
   {output}
   ```
3. **Loss masking** — the prompt portion and the response portion are tokenized
   separately, then concatenated. The `labels` array is set to `-100` (PyTorch's
   `ignore_index` convention) over every prompt-token position, so cross-entropy loss —
   and therefore gradient — is only computed on the response tokens. Without this, the
   model would spend training signal re-learning to reproduce the fixed prompt template
   itself, rather than learning to produce good responses.
4. **Batching** — variable-length examples are padded to a common length per batch
   (`collate_batch`); padding positions get `attention_mask = 0` (ignored by attention) and
   `labels = -100` (ignored by loss), for the same reason as prompt masking above.
5. **Training loop** — standard forward → `loss.backward()` (autodiff computes every
   weight's gradient) → `torch.nn.utils.clip_grad_norm_` (clips exploding gradients) → 
   `optimizer.step()` (AdamW nudges every weight against its gradient) → `scheduler.step()` 
   (cosine decay with warmup) → `optimizer.zero_grad()` (clears gradients so they don't 
   accumulate into the next batch). Runs on Apple's `mps` backend automatically when available.
6. **Export** — reuses the exact same `named_parameters()` → flatten → `.txt` loop from
   `main.py`, writing to `../weights_finetuned/` so the base weights are never overwritten.

The architecture is identical before and after fine-tuning — only the numeric weight
values change — which is why the same C++ inference engine reads both without
modification.

---

## Build & run inference

```bash
cd inference
g++ -std=c++20 -O3 -march=native main.cc -o gpt2

# single prompt
./gpt2 "The meaning of life is" 60 0.8 40 0.9 1.2

# use fine-tuned weights
./gpt2 "### Instruction:
explain gravity

### Response:
" 80 0.7 40 0.9 1.1 --weights ../weights_finetuned

# interactive mode (launches automatically after the optional one-shot prompt)
./gpt2
```

Positional arguments: `prompt  tokens  temperature  topK  topP  repetitionPenalty`.
`temperature 0` → greedy decoding. All fields have sensible defaults. `--weights <path>`
can appear anywhere in the argument list.

In interactive mode, every field (`tokens`, `topP`, `repetitionPenalty`) is editable per
prompt — press Enter to keep the current default.

---

## Export weights from HuggingFace

```bash
cd modelLoader
pip install transformers torch

python main.py
```

Dumps all weight tensors to `../weights/` as space-separated float32 `.txt` files,
alongside `vocab.json` and `merges.txt`. 

To drastically speed up C++ load times (10-30x faster), convert these `.txt` files to raw
`.bin` format:
```bash
python convert_to_bin.py
```
The C++ loader will automatically pick up the `.bin` files instead of the `.txt` files.

---

## Fine-tuning on Alpaca (SFT)

```bash
cd modelLoader
pip install datasets
python train_sft.py
```

Trains on a configurable subset of Alpaca (default 1000 examples, 3 epochs),
masking loss to the response portion only, then exports to `../weights_finetuned/`.
Uses Apple MPS automatically when available. `vocab.json`/`merges.txt` are unchanged by
fine-tuning and are copied over from `../weights/` rather than re-exported.

---

## Inference features

- **KV cache** — prompt tokens processed once; only the new token runs the full forward
  pass per generation step.
- **Sampling** — top-k, top-p (nucleus), temperature, repetition penalty.
- **Tied LM head** — the embedding matrix is reused for the output logit projection.
- **Binary weights** — `.bin` fast path over `.txt` parsing.
- **Streaming output** — safe against multi-byte UTF-8 characters split across token
  boundaries.

---

## Model config

| Parameter | Value |
|-----------|-------|
| Layers | 12 |
| Embedding dim | 768 |
| Attention heads | 12 |
| Head dim | 64 |
| MLP dim | 3072 |
| Vocabulary | 50257 |
| Context length | 1024 |