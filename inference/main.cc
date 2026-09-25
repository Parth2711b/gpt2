// GPT-2 (124M) inference — C++20, no ML framework.
//
// Build:  g++ -std=c++20 -O3 -march=native main.cc -o gpt2
// Run:    ./gpt2 [prompt] [tokens] [temp] [topK] [topP] [repPenalty] [--weights
// <path>]
//         temp 0 = greedy.  Defaults: 20 tokens, 0.8 temp, topK 40, topP 0.9,
//         penalty 1.2.
//         --weights can appear anywhere; default path is ../weights.
//
// Weight files expected in the weights directory (txt or matching .bin):
//   transformer.wte.weight, transformer.wpe.weight,
//   transformer.ln_f.{weight,bias}
//   transformer.h.N.{ln_1,ln_2,attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}.{weight,bias}
//   vocab.json, merges.txt

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>

#include <chrono>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using tensor = std::vector<float>;

constexpr int kLayers = 12;
constexpr int kEmbeddingDim = 768;
constexpr int kHeads = 12;
constexpr int kHeadDim = kEmbeddingDim / kHeads;
constexpr int kMlpDim = 3072;
constexpr int kVocabularySize = 50257;
constexpr int kMaxPositions = 1024;
constexpr int kEndOfText = 50256;
constexpr float kEpsilon = 1e-5f;

// ---------- Weights ----------

struct TransformerInput {
  tensor qWeights, kWeights, vWeights;
  tensor qBiases, kBiases, vBiases;
  tensor l1Weights, l2Weights;
  tensor l1Biases, l2Biases;
  tensor oWeights, oBiases;
  tensor lnAttWeights, lnAttBiases;
  tensor lnMlpWeights, lnMlpBiases;
};

struct GptWeights {
  tensor embeddingWeights; // wte: [vocab][768], also used as tied lm_head
  tensor positionalEmbeddingWeights; // wpe: [1024][768]
  std::vector<TransformerInput> transformerWeights;
  tensor finalNormWeights, finalNormBiases;

  GptWeights() : transformerWeights(kLayers) {}
};

bool endsWith(const std::string &text, const std::string &suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Loads a matching .bin (raw float32) if present, otherwise parses the .txt
// file.
tensor readValues(const std::string &path, std::size_t maxCount = 0) {
  if (endsWith(path, ".txt")) {
    const std::string binPath = path.substr(0, path.size() - 4) + ".bin";
    std::ifstream bin(binPath, std::ios::binary | std::ios::ate);
    if (bin) {
      const std::streamoff bytes = bin.tellg();
      if (bytes < 0 ||
          bytes % static_cast<std::streamoff>(sizeof(float)) != 0) {
        throw std::runtime_error("Corrupt binary file: " + binPath);
      }
      const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(float);
      if (maxCount != 0 && count != maxCount) {
        throw std::runtime_error("Wrong value count in: " + binPath);
      }
      tensor values(count);
      bin.seekg(0);
      bin.read(reinterpret_cast<char *>(values.data()),
               static_cast<std::streamsize>(bytes));
      if (!bin)
        throw std::runtime_error("Could not read: " + binPath);
      return values;
    }
  }

  std::ifstream stream(path);
  if (!stream)
    throw std::runtime_error("Could not open weights file: " + path);

  tensor values;
  if (maxCount != 0)
    values.reserve(maxCount);
  float value = 0.0f;
  while ((maxCount == 0 || values.size() < maxCount) && stream >> value) {
    values.push_back(value);
  }
  if (maxCount != 0) {
    if (values.size() != maxCount)
      throw std::runtime_error("Too few values in: " + path);
    if (stream >> value)
      throw std::runtime_error("Too many values (wrong shape?) in: " + path);
  }
  return values;
}

// HF Conv1D stores weights as [in][out]; transpose to [out][in] so forwardPass
// rows are neurons.
tensor transposeMatrix(const tensor &values, int rows, int cols) {
  assert(values.size() == static_cast<std::size_t>(rows) * cols);
  tensor result(values.size());
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      result[static_cast<std::size_t>(col) * rows + row] =
          values[static_cast<std::size_t>(row) * cols + col];
    }
  }
  return result;
}

tensor loadMatrix(const std::string &path, int inputDim, int outputDim) {
  return transposeMatrix(
      readValues(path, static_cast<std::size_t>(inputDim) * outputDim),
      inputDim, outputDim);
}

void loadTransformerLayer(TransformerInput &layer, int index,
                          const std::string &root) {
  const std::string prefix = root + "/transformer.h." + std::to_string(index);
  constexpr std::ptrdiff_t block =
      static_cast<std::ptrdiff_t>(kEmbeddingDim) * kEmbeddingDim;

  // c_attn is fused: after transpose, rows are [q (768) | k (768) | v (768)]
  tensor fusedWeights = loadMatrix(prefix + ".attn.c_attn.weight.txt",
                                   kEmbeddingDim, 3 * kEmbeddingDim);
  tensor fusedBiases =
      readValues(prefix + ".attn.c_attn.bias.txt", 3 * kEmbeddingDim);

  layer.qWeights.assign(fusedWeights.begin(), fusedWeights.begin() + block);
  layer.kWeights.assign(fusedWeights.begin() + block,
                        fusedWeights.begin() + 2 * block);
  layer.vWeights.assign(fusedWeights.begin() + 2 * block, fusedWeights.end());
  layer.qBiases.assign(fusedBiases.begin(),
                       fusedBiases.begin() + kEmbeddingDim);
  layer.kBiases.assign(fusedBiases.begin() + kEmbeddingDim,
                       fusedBiases.begin() + 2 * kEmbeddingDim);
  layer.vBiases.assign(fusedBiases.begin() + 2 * kEmbeddingDim,
                       fusedBiases.end());

  layer.oWeights = loadMatrix(prefix + ".attn.c_proj.weight.txt", kEmbeddingDim,
                              kEmbeddingDim);
  layer.oBiases = readValues(prefix + ".attn.c_proj.bias.txt", kEmbeddingDim);
  layer.l1Weights =
      loadMatrix(prefix + ".mlp.c_fc.weight.txt", kEmbeddingDim, kMlpDim);
  layer.l1Biases = readValues(prefix + ".mlp.c_fc.bias.txt", kMlpDim);
  layer.l2Weights =
      loadMatrix(prefix + ".mlp.c_proj.weight.txt", kMlpDim, kEmbeddingDim);
  layer.l2Biases = readValues(prefix + ".mlp.c_proj.bias.txt", kEmbeddingDim);
  layer.lnAttWeights = readValues(prefix + ".ln_1.weight.txt", kEmbeddingDim);
  layer.lnAttBiases = readValues(prefix + ".ln_1.bias.txt", kEmbeddingDim);
  layer.lnMlpWeights = readValues(prefix + ".ln_2.weight.txt", kEmbeddingDim);
  layer.lnMlpBiases = readValues(prefix + ".ln_2.bias.txt", kEmbeddingDim);
}

GptWeights loadWeights(const std::string &root = "../weights") {
  GptWeights weights;
  weights.embeddingWeights =
      readValues(root + "/transformer.wte.weight.txt",
                 static_cast<std::size_t>(kVocabularySize) * kEmbeddingDim);
  weights.positionalEmbeddingWeights =
      readValues(root + "/transformer.wpe.weight.txt",
                 static_cast<std::size_t>(kMaxPositions) * kEmbeddingDim);
  weights.finalNormWeights =
      readValues(root + "/transformer.ln_f.weight.txt", kEmbeddingDim);
  weights.finalNormBiases =
      readValues(root + "/transformer.ln_f.bias.txt", kEmbeddingDim);

  for (int layer = 0; layer < kLayers; ++layer) {
    loadTransformerLayer(weights.transformerWeights[layer], layer, root);
  }
  return weights;
}

// ---------- Math ----------

float gelu(float value) { // GPT-2 "gelu_new" tanh approximation
  constexpr float kSqrt2OverPi = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + std::tanh(kSqrt2OverPi *
                           (value + 0.044715f * value * value * value)));
}

tensor softmax(const tensor &input) {
  if (input.empty())
    return {};
  const float maximum = *std::max_element(input.begin(), input.end());
  tensor result(input.size());
  for (std::size_t i = 0; i < input.size(); ++i)
    result[i] = std::exp(input[i] - maximum);
  const float sum = std::accumulate(result.begin(), result.end(), 0.0f);
  for (float &value : result)
    value /= sum;
  return result;
}

// weights layout: [out][in]
tensor forwardPass(const tensor &input, const tensor &weights,
                   const tensor &biases, bool useGelu = false) {
  const std::size_t outputSize = biases.size();
  assert(weights.size() == input.size() * outputSize);
  tensor output(biases.begin(), biases.end());
  for (std::size_t out = 0; out < outputSize; ++out) {
    const float *row = weights.data() + out * input.size();
    float sum = output[out];
    for (std::size_t in = 0; in < input.size(); ++in)
      sum += input[in] * row[in];
    output[out] = useGelu ? gelu(sum) : sum;
  }
  return output;
}

tensor layerNorm(const tensor &input, const tensor &weights,
                 const tensor &biases) {
  assert(input.size() == weights.size() && input.size() == biases.size());
  const float mean =
      std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
  float variance = 0.0f;
  for (float value : input)
    variance += (value - mean) * (value - mean);
  variance /= input.size();

  tensor output(input.size());
  const float scale = std::sqrt(variance + kEpsilon);
  for (std::size_t i = 0; i < input.size(); ++i) {
    output[i] = ((input[i] - mean) / scale) * weights[i] + biases[i];
  }
  return output;
}

tensor addVectors(const tensor &left, const tensor &right) {
  assert(left.size() == right.size());
  tensor result(left.begin(), left.end());
  for (std::size_t i = 0; i < result.size(); ++i)
    result[i] += right[i];
  return result;
}

// ---------- Transformer (with KV cache) ----------

struct KvCache {
  std::vector<tensor> keys, values; // per layer, flattened [token][768]
  KvCache() : keys(kLayers), values(kLayers) {
    for (int layer = 0; layer < kLayers; ++layer) {
      keys[layer].reserve(static_cast<std::size_t>(kMaxPositions) *
                          kEmbeddingDim);
      values[layer].reserve(static_cast<std::size_t>(kMaxPositions) *
                            kEmbeddingDim);
    }
  }
};

// One transformer layer on a single new token x: x -> x + Attn(LN1(x)) -> +
// MLP(LN2(.)) Attention runs against the cached K/V from all previous tokens
// (causal, via the cache order).
tensor transformerBlockStep(const TransformerInput &layer,
                            const tensor &x, KvCache &cache,
                            int layerIndex) {
  tensor normed = layerNorm(x, layer.lnAttWeights, layer.lnAttBiases);
  tensor q = forwardPass(normed, layer.qWeights, layer.qBiases);
  tensor k = forwardPass(normed, layer.kWeights, layer.kBiases);
  tensor v = forwardPass(normed, layer.vWeights, layer.vBiases);

  tensor &cachedKeys = cache.keys[layerIndex];
  tensor &cachedValues = cache.values[layerIndex];
  cachedKeys.insert(cachedKeys.end(), k.begin(), k.end());
  cachedValues.insert(cachedValues.end(), v.begin(), v.end());

  const int count = static_cast<int>(cachedKeys.size() / kEmbeddingDim);
  const float scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  tensor attended(kEmbeddingDim, 0.0f);
  for (int head = 0; head < kHeads; ++head) {
    const std::size_t headOffset = static_cast<std::size_t>(head) * kHeadDim;
    tensor scores(count);
    for (int t = 0; t < count; ++t) {
      const float *key = cachedKeys.data() +
                         static_cast<std::size_t>(t) * kEmbeddingDim +
                         headOffset;
      scores[t] =
          std::inner_product(q.begin() + headOffset,
                             q.begin() + headOffset + kHeadDim, key, 0.0f) *
          scale;
    }
    tensor probabilities = softmax(scores);
    for (int t = 0; t < count; ++t) {
      const float *value = cachedValues.data() +
                           static_cast<std::size_t>(t) * kEmbeddingDim +
                           headOffset;
      for (int d = 0; d < kHeadDim; ++d)
        attended[headOffset + d] += probabilities[t] * value[d];
    }
  }

  tensor projected = forwardPass(attended, layer.oWeights, layer.oBiases);
  tensor residual = addVectors(x, projected);

  tensor normed2 = layerNorm(residual, layer.lnMlpWeights, layer.lnMlpBiases);
  tensor hidden = forwardPass(normed2, layer.l1Weights, layer.l1Biases, true);
  tensor mlpOutput = forwardPass(hidden, layer.l2Weights, layer.l2Biases);
  return addVectors(residual, mlpOutput);
}

// Runs one token through the full model. computeLogits=false skips the final
// layernorm and the 50257-way logits (used for prompt tokens other than the
// last one, to save work).
tensor gptStep(const GptWeights &weights, int tokenId, int position,
               KvCache &cache, bool computeLogits) {
  if (position < 0 || position >= kMaxPositions) {
    throw std::runtime_error("Position out of range (max 1024 tokens)");
  }
  tensor x(kEmbeddingDim);
  const std::size_t tokenBase =
      static_cast<std::size_t>(tokenId) * kEmbeddingDim;
  const std::size_t positionBase =
      static_cast<std::size_t>(position) * kEmbeddingDim;
  for (int d = 0; d < kEmbeddingDim; ++d) {
    x[d] = weights.embeddingWeights[tokenBase + d] +
           weights.positionalEmbeddingWeights[positionBase + d];
  }

  for (int layer = 0; layer < kLayers; ++layer) {
    x = transformerBlockStep(weights.transformerWeights[layer], x, cache,
                             layer);
  }
  if (!computeLogits)
    return {};

  tensor normalized =
      layerNorm(x, weights.finalNormWeights, weights.finalNormBiases);
  tensor logits(
      kVocabularySize); // tied lm_head: logits = normalized . wte[token]
  for (int token = 0; token < kVocabularySize; ++token) {
    const float *wordEmbedding = weights.embeddingWeights.data()
        + static_cast<std::size_t>(token) * kEmbeddingDim;
    logits[token] = std::inner_product(normalized.begin(), normalized.end(),
                                       wordEmbedding, 0.0f);
  }
  return logits;
}

// ---------- Sampling ----------

int argmaxToken(const tensor &logits) {
  return static_cast<int>(std::distance(
      logits.begin(), std::max_element(logits.begin(), logits.end())));
}

// HF-style repetition penalty: logits for tokens already seen are pulled toward
// zero (positive / penalty, negative * penalty). penalty <= 1.0 is a no-op.
void applyRepetitionPenalty(tensor &logits, const std::vector<int> &seenTokens,
                            float penalty) {
  if (penalty <= 1.0f)
    return;
  std::unordered_set<int> seen(seenTokens.begin(), seenTokens.end());
  for (int id : seen) {
    float &value = logits[id];
    value = value > 0.0f ? value / penalty : value * penalty;
  }
}

// Top-k candidates, then a nucleus (top-p) cutoff on that sorted set, then
// sample.
int sampleToken(const tensor &logits, float temperature, int topK, float topP,
                std::mt19937 &rng) {
  std::vector<int> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);
  topK = std::clamp(topK, 1, static_cast<int>(indices.size()));
  std::partial_sort(indices.begin(), indices.begin() + topK, indices.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  // indices[0..topK) are sorted descending by logit, so probabilities[] below
  // is too.

  tensor top(topK);
  for (int i = 0; i < topK; ++i)
    top[i] = logits[indices[i]] / temperature;
  tensor probabilities = softmax(top);

  int keep = topK;
  if (topP < 1.0f) {
    float cumulative = 0.0f;
    keep = 0;
    for (int i = 0; i < topK; ++i) {
      cumulative += probabilities[i];
      ++keep;
      if (cumulative >= topP)
        break;
    }
  }

  tensor kept(probabilities.begin(), probabilities.begin() + keep);
  const float sum = std::accumulate(kept.begin(), kept.end(), 0.0f);
  for (float &p : kept)
    p /= sum;

  std::discrete_distribution<int> distribution(kept.begin(), kept.end());
  return indices[distribution(rng)];
}

// ---------- Tokenizer (GPT-2 byte-level BPE) ----------

std::vector<std::string> gpt2Tokens; // id -> token string
std::unordered_map<std::string, int> tokenToId;
std::unordered_map<std::string, int>
    mergeRanks; // "left right" -> rank (lower merges first)
std::array<std::string, 256> byteToUnicode;
std::unordered_map<char32_t, unsigned char> unicodeToByte;

std::string codepointToUtf8(char32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else if (cp < 0x10000) {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xF0 | (cp >> 18));
    out += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

std::vector<char32_t> utf8ToCodepoints(const std::string &text) {
  std::vector<char32_t> result;
  for (std::size_t i = 0; i < text.size();) {
    const unsigned char first = static_cast<unsigned char>(text[i]);
    const int length = first < 0x80          ? 1
                       : (first >> 5) == 0x6 ? 2
                       : (first >> 4) == 0xE ? 3
                                             : 4;
    char32_t cp = length == 1 ? first : (first & (0xFF >> (length + 1)));
    for (int k = 1; k < length && i + k < text.size(); ++k) {
      cp = (cp << 6) | (static_cast<unsigned char>(text[i + k]) & 0x3F);
    }
    result.push_back(cp);
    i += length;
  }
  return result;
}

// GPT-2's bytes_to_unicode(): printable bytes map to themselves, the rest to a
// spare range.
void buildByteTables() {
  std::array<bool, 256> printable{};
  for (int b = 33; b <= 126; ++b)
    printable[b] = true;
  for (int b = 161; b <= 172; ++b)
    printable[b] = true;
  for (int b = 174; b <= 255; ++b)
    printable[b] = true;

  int extra = 0;
  for (int b = 0; b < 256; ++b) {
    const char32_t cp = printable[b] ? static_cast<char32_t>(b)
                                     : static_cast<char32_t>(256 + extra++);
    byteToUnicode[b] = codepointToUtf8(cp);
    unicodeToByte[cp] = static_cast<unsigned char>(b);
  }
}

bool isSpaceByte(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
         c == '\v';
}
bool isDigitByte(unsigned char c) { return c >= '0' && c <= '9'; }
bool isLetterByte(unsigned char c) { // non-ASCII bytes are treated as letters
                                     // (\p{L} approximation)
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80;
}

// Manual version of GPT-2's pre-tokenizer regex (std::regex has no \p{L}
// support): 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
std::vector<std::string> preTokenize(const std::string &text) {
  static constexpr std::array<std::string_view, 7> contractions = {
      "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"};
  const std::size_t n = text.size();
  std::vector<std::string> pieces;

  auto byteAt = [&](std::size_t i) {
    return static_cast<unsigned char>(text[i]);
  };
  auto runEnd = [&](std::size_t start) {
    std::size_t end = start;
    const unsigned char first = byteAt(start);
    if (isLetterByte(first)) {
      while (end < n && isLetterByte(byteAt(end)))
        ++end;
    } else if (isDigitByte(first)) {
      while (end < n && isDigitByte(byteAt(end)))
        ++end;
    } else {
      while (end < n && !isSpaceByte(byteAt(end)) &&
             !isLetterByte(byteAt(end)) && !isDigitByte(byteAt(end)))
        ++end;
    }
    return end;
  };

  std::size_t i = 0;
  while (i < n) {
    bool matched = false;
    if (text[i] == '\'') {
      for (std::string_view contraction : contractions) {
        if (text.compare(i, contraction.size(), contraction) == 0) {
          pieces.emplace_back(contraction);
          i += contraction.size();
          matched = true;
          break;
        }
      }
      if (matched)
        continue;
    }

    if (text[i] == ' ' && i + 1 < n && !isSpaceByte(byteAt(i + 1))) {
      const std::size_t end = runEnd(i + 1);
      pieces.push_back(text.substr(i, end - i));
      i = end;
    } else if (!isSpaceByte(byteAt(i))) {
      const std::size_t end = runEnd(i);
      pieces.push_back(text.substr(i, end - i));
      i = end;
    } else {
      // whitespace run: \s+(?!\S) keeps the last whitespace char for the next
      // token
      std::size_t end = i;
      while (end < n && isSpaceByte(byteAt(end)))
        ++end;
      if (end < n && end - i > 1)
        --end;
      pieces.push_back(text.substr(i, end - i));
      i = end;
    }
  }
  return pieces;
}

std::vector<int> bpeEncode(const std::string &piece) {
  std::vector<std::string> symbols;
  symbols.reserve(piece.size());
  for (unsigned char byte : piece)
    symbols.push_back(byteToUnicode[byte]);

  while (symbols.size() > 1) {
    int bestRank = std::numeric_limits<int>::max();
    std::size_t bestIndex = symbols.size();
    for (std::size_t i = 0; i + 1 < symbols.size(); ++i) {
      auto it = mergeRanks.find(symbols[i] + " " + symbols[i + 1]);
      if (it != mergeRanks.end() && it->second < bestRank) {
        bestRank = it->second;
        bestIndex = i;
      }
    }
    if (bestIndex == symbols.size())
      break;

    const std::string first = symbols[bestIndex];
    const std::string second = symbols[bestIndex + 1];
    std::vector<std::string> merged;
    merged.reserve(symbols.size());
    for (std::size_t i = 0; i < symbols.size();) {
      if (i + 1 < symbols.size() && symbols[i] == first &&
          symbols[i + 1] == second) {
        merged.push_back(first + second);
        i += 2;
      } else {
        merged.push_back(symbols[i]);
        ++i;
      }
    }
    symbols = std::move(merged);
  }

  std::vector<int> ids;
  ids.reserve(symbols.size());
  for (const std::string &symbol : symbols) {
    auto it = tokenToId.find(symbol);
    if (it == tokenToId.end())
      throw std::runtime_error("Token not in vocab: " + symbol);
    ids.push_back(it->second);
  }
  return ids;
}

std::vector<int> encode(const std::string &text) {
  std::vector<int> ids;
  for (const std::string &piece : preTokenize(text)) {
    for (int id : bpeEncode(piece))
      ids.push_back(id);
  }
  return ids;
}

// Number of trailing bytes forming an incomplete multi-byte UTF-8 sequence (0 =
// complete). Used to avoid printing a character split across two token
// boundaries mid-stream.
std::size_t incompleteUtf8TailLength(const std::string &buffer) {
  const std::size_t limit = std::min<std::size_t>(4, buffer.size());
  for (std::size_t back = 1; back <= limit; ++back) {
    const unsigned char b =
        static_cast<unsigned char>(buffer[buffer.size() - back]);
    if ((b & 0xC0) == 0x80)
      continue; // continuation byte, keep looking back
    int length = 1;
    if ((b & 0xE0) == 0xC0)
      length = 2;
    else if ((b & 0xF0) == 0xE0)
      length = 3;
    else if ((b & 0xF8) == 0xF0)
      length = 4;
    return back < static_cast<std::size_t>(length) ? back : 0;
  }
  return 0;
}

std::string decodeToken(int id) {
  std::string bytes;
  for (char32_t cp : utf8ToCodepoints(gpt2Tokens.at(id))) {
    bytes += static_cast<char>(unicodeToByte.at(cp));
  }
  return bytes;
}

// vocab.json is a flat {"token": id, ...} object; parsed manually to avoid a
// JSON dependency.
void loadVocab(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Could not open vocab file: " + path);
  const std::string text((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());

  gpt2Tokens.assign(kVocabularySize, "");
  tokenToId.clear();
  tokenToId.reserve(kVocabularySize * 2);

  std::size_t pos = 0;
  auto skipSpaces = [&] {
    while (pos < text.size() &&
           isSpaceByte(static_cast<unsigned char>(text[pos])))
      ++pos;
  };
  auto expect = [&](char c) {
    skipSpaces();
    if (pos >= text.size() || text[pos] != c) {
      throw std::runtime_error(
          std::string("vocab.json parse error, expected '") + c + "'");
    }
    ++pos;
  };
  auto parseString = [&]() {
    expect('"');
    std::string out;
    while (pos < text.size() && text[pos] != '"') {
      const char c = text[pos++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (pos >= text.size())
        break;
      const char escape = text[pos++];
      switch (escape) {
      case 'n':
        out += '\n';
        break;
      case 't':
        out += '\t';
        break;
      case 'r':
        out += '\r';
        break;
      case 'b':
        out += '\b';
        break;
      case 'f':
        out += '\f';
        break;
      case '"':
      case '\\':
      case '/':
        out += escape;
        break;
      case 'u':
        out += codepointToUtf8(static_cast<char32_t>(
            std::stoul(text.substr(pos, 4), nullptr, 16)));
        pos += 4;
        break;
      default:
        throw std::runtime_error("vocab.json: bad escape");
      }
    }
    ++pos;
    return out;
  };

  expect('{');
  skipSpaces();
  while (pos < text.size() && text[pos] != '}') {
    std::string token = parseString();
    expect(':');
    skipSpaces();
    int id = 0;
    while (pos < text.size() &&
           isDigitByte(static_cast<unsigned char>(text[pos]))) {
      id = id * 10 + (text[pos++] - '0');
    }
    if (id < 0 || id >= kVocabularySize)
      throw std::runtime_error("vocab.json: id out of range");
    gpt2Tokens[id] = token;
    tokenToId.emplace(std::move(token), id);
    skipSpaces();
    if (pos < text.size() && text[pos] == ',')
      ++pos;
    skipSpaces();
  }
}

// merges.txt: each line is "left right"; line number is the rank. First line
// may be a header.
void loadMerges(const std::string &path) {
  std::ifstream file(path);
  if (!file)
    throw std::runtime_error("Could not open merges file: " + path);
  mergeRanks.clear();
  std::string line;
  int rank = 0;
  bool firstLine = true;
  while (std::getline(file, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (firstLine) {
      firstLine = false;
      if (line.rfind("#version", 0) == 0)
        continue;
    }
    if (line.empty())
      continue;
    mergeRanks.emplace(line, rank++);
  }
}

// ---------- Generation ----------

// temperature <= 0 gives greedy decoding, otherwise top-k + top-p sampling.
// repetitionPenalty > 1.0 suppresses logits for tokens already in the sequence.
std::string generate(const GptWeights &weights, const std::string &prompt,
                     int maxNewTokens, float temperature, int topK, float topP,
                     float repetitionPenalty) {
  std::vector<int> ids = encode(prompt);
  if (ids.empty())
    ids.push_back(kEndOfText);
  if (ids.size() >= static_cast<std::size_t>(kMaxPositions)) {
    throw std::runtime_error("Prompt too long (max 1023 tokens)");
  }

  KvCache cache;
  std::mt19937 rng(std::random_device{}());

  tensor logits;
  for (std::size_t i = 0; i < ids.size(); ++i) {
    logits = gptStep(weights, ids[i], static_cast<int>(i), cache,
                     i + 1 == ids.size());
  }

  std::string generated;
  std::string pendingUtf8;
  std::cout << prompt << std::flush;

  int tokensGenerated = 0;
  const auto t0 = std::chrono::steady_clock::now();

  for (int step = 0; step < maxNewTokens; ++step) {
    applyRepetitionPenalty(logits, ids, repetitionPenalty);
    const int next = temperature <= 0.0f
                         ? argmaxToken(logits)
                         : sampleToken(logits, temperature, topK, topP, rng);
    if (next == kEndOfText)
      break;
    ids.push_back(next);
    ++tokensGenerated;

    const std::string piece = decodeToken(next);
    generated += piece;
    pendingUtf8 += piece;
    const std::size_t incomplete = incompleteUtf8TailLength(pendingUtf8);
    const std::size_t printable = pendingUtf8.size() - incomplete;
    std::cout << pendingUtf8.substr(0, printable) << std::flush;
    pendingUtf8 = pendingUtf8.substr(printable);

    if (step + 1 == maxNewTokens ||
        ids.size() >= static_cast<std::size_t>(kMaxPositions))
      break;
    logits =
        gptStep(weights, next, static_cast<int>(ids.size()) - 1, cache, true);
  }
  std::cout << pendingUtf8 << '\n';

  if (tokensGenerated > 0) {
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::cerr << "[" << tokensGenerated << " tokens, "
              << std::fixed << std::setprecision(1)
              << (tokensGenerated / secs) << " tok/s]\n";
  }

  return generated;
}

int readIntOrDefault(const std::string &label, int fallback) {
  std::cout << label << " [" << fallback << "]: " << std::flush;
  std::string line;
  if (!std::getline(std::cin, line) || line.empty())
    return fallback;
  try {
    return std::stoi(line);
  } catch (const std::exception &) {
    std::cerr << "  invalid input, using default: " << fallback << "\n";
    return fallback;
  }
}

float readFloatOrDefault(const std::string &label, float fallback) {
  std::cout << label << " [" << fallback << "]: " << std::flush;
  std::string line;
  if (!std::getline(std::cin, line) || line.empty())
    return fallback;
  try {
    return std::stof(line);
  } catch (const std::exception &) {
    std::cerr << "  invalid input, using default: " << fallback << "\n";
    return fallback;
  }
}

int main(int argc, char **argv) {
  try {
    // Pull out an optional "--weights <path>" flag first, wherever it appears,
    // leaving the rest as ordinary positional args (prompt, tokens, temp, ...).
    std::string root = "../weights";
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
      if (std::string(argv[i]) == "--weights" && i + 1 < argc) {
        root = argv[++i];
      } else {
        positional.push_back(argv[i]);
      }
    }
    // --help
    if (positional.size() == 1 &&
        (positional[0] == "--help" || positional[0] == "-h")) {
      std::cout
          << "Usage: ./gpt2 [prompt] [tokens] [temp] [topK] [topP] "
             "[repPenalty] [--weights <path>]\n"
          << "  prompt          text to continue (default: interactive mode)\n"
          << "  tokens          max new tokens to generate (default: 20)\n"
          << "  temp            temperature; 0 = greedy (default: 0.8)\n"
          << "  topK            top-k candidates (default: 40)\n"
          << "  topP            nucleus probability cutoff (default: 0.9)\n"
          << "  repPenalty      repetition penalty >= 1.0 (default: 1.2)\n"
          << "  --weights <dir> weights directory (default: ../weights)\n"
          << "\nExample:\n"
          << "  ./gpt2 \"The meaning of life is\" 60 0.8 40 0.9 1.2\n"
          << "  ./gpt2 --weights ../weights_finetuned\n";
      return 0;
    }

    const int maxNewTokens =
        positional.size() > 1 ? std::stoi(positional[1]) : 20;

    const float temperature =
        positional.size() > 2 ? std::stof(positional[2]) : 0.8f;
    const int topK = positional.size() > 3 ? std::stoi(positional[3]) : 40;
    const float topP = positional.size() > 4 ? std::stof(positional[4]) : 0.9f;
    const float repetitionPenalty =
        positional.size() > 5 ? std::stof(positional[5]) : 1.2f;

    buildByteTables();
    loadVocab(root + "/vocab.json");
    loadMerges(root + "/merges.txt");

    std::cerr << "Loading weights from " << root << "...\n";
    GptWeights weights =
        loadWeights(root); // loaded once, reused for every prompt below
    std::cerr << "Loaded GPT-2 weights: " << weights.transformerWeights.size()
              << " layers\n";

    if (!positional.empty()) {
      generate(weights, positional[0], maxNewTokens, temperature, topK, topP,
               repetitionPenalty);
    }

    std::cout << "\nInteractive mode — leave any field blank to keep the "
                 "current default.\n"
                 "Type 'exit' or an empty prompt to quit.\n";

    while (true) {
      std::cout << "\nprompt: " << std::flush;
      std::string prompt;
      if (!std::getline(std::cin, prompt))
        break;
      if (prompt.empty() || prompt == "exit" || prompt == "quit")
        break;

      const int roundTokens = readIntOrDefault("tokens", maxNewTokens);
      const float roundTemperature =
          readFloatOrDefault("temperature", temperature);
      const int roundTopK = readIntOrDefault("topK", topK);
      const float roundTopP = readFloatOrDefault("topP", topP);
      const float roundRepetitionPenalty =
          readFloatOrDefault("repetitionPenalty", repetitionPenalty);

      generate(weights, prompt, roundTokens, roundTemperature, roundTopK,
               roundTopP, roundRepetitionPenalty);
    }
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}