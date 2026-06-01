# 🧠 LuaGPT-1: Implementacja GPT-1 od Zera w Roblox Studio

> **PL** | [EN below](#english-version)

---

## 📌 Opis projektu

LuaGPT-1 to w pelni dzialajaca implementacja architektury GPT-1 napisana w jezyku **Lua**, zbudowana od zera wewnatrz srodowiska **Roblox Studio** — bez zadnych zewnetrznych bibliotek ML.

Projekt zawiera kompletny pipeline: tokenizacje tekstu, embeddingi tokenow, positional encoding, wielowarstwowy mechanizm self-attention z maskowaniem kauzalnym, siec feed-forward z sigmoid, LM Head z softmaxem oraz recznie zaimplementowana wsteczna propagacje bledow.

---

## 🗂 Struktura projektu

```
LuaGPT-1/
├── api.lua          -- Glowny punkt wejscia: trening + generowanie tekstu
├── Tokenizer.lua    -- Tokenizacja, budowa slownika, encode/decode
├── Transformer.lua  -- Self-Attention, Feed-Forward, Forward pass
├── Trainer.lua      -- Petla treningowa, forward pass, backpropagation
└── data.lua         -- Konfiguracja: embeddingi, wagi, slownik, warstwy
```

---

## ⚙️ Architektura

### Parametry modelu

| Parametr | Wartosc |
|---|---|
| Wymiar embeddingu (D) | **16** |
| Liczba warstw attention | **12** |
| Liczba glowic (heads) | **12** |
| Dlugosc kontekstu | **4 tokeny** |
| Rozmiar slownika | dynamiczny (budowany z danych) |
| Warstwy FFN | **2** (16→32→16 z sigmoid) |
| Kara za powtorzenia | **1.45** (exponential penalty) |
| Temperature | **0.8** |

### Schemat przepływu danych

```
Wejscie (tekst)
      ↓
Tokenizer.Encode()         -- lowercase + czyszczenie + mapowanie na ID
      ↓
Token Embeddings           -- tabela D=16 wektorow per token
      ↓
[Self-Attention × 12]      -- Q/K/V projections + causal masking + residual
      ↓
Feed-Forward (sigmoid)     -- 16 → 32 → 16
      ↓
LM Head (dot product)      -- last_vector · weights[w_id] + bias
      ↓
Softmax + Temperature      -- normalizacja prawdopodobienstw
      ↓
Weighted Sampling          -- losowanie z rozkladu
      ↓
Tokenizer.Decode()         -- mapowanie ID → slowa
```

---

## 🔍 Szczegoly implementacji

### Tokenizer (`Tokenizer.lua`)

- Buduje slownik dynamicznie z tekstu treningowego (`BuildVocab`)
- Czyszczenie: lowercase, usuniecie interpunkcji, obsluga kropek jako osobnych tokenow
- `Encode` / `Decode` — pelna obsluga sekwencji
- Cache zero-wektora zamiast alokacji przy kazdym wywolaniu (`ZERO_VECTOR`)
- Zachowuje wytrenowane wagi jesli `EmbeddingTable` juz istnieje

### Self-Attention (`Transformer.lua`)

- Projekcje **Q, K, V, O** jako wektory skalarne per wymiar (uproszczona implementacja)
- **Causal masking** — token `i` widzi tylko tokeny `j ≤ i`
- Skalowanie przez `1/sqrt(D)`
- Numeryczna stabilnosc: `math.clamp(dot * scale, -20, 20)` przed `math.exp`
- **Residual connection**: `seq[i] += attn_out[i]` po kazdej warstwie
- 12 niezaleznych warstw (`v1`..`v12`) z osobnymi wagami `Wq/Wk/Wv/Wo/Bq/Bk/Bv/Bo`

### Feed-Forward (`Transformer.lua`)

- 2 warstwy: `16 → 32 → 16`
- Funkcja aktywacji: **Sigmoid** (`1 / (1 + exp(-x))`)
- Wagi inicjalizowane przez `sin/cos` dla deterministycznosci

### Backpropagation (`Trainer.lua`)

- Recznie napisany gradient przez **cross-entropy loss**
- Gradient logitow: `d_logit = prob - 1` dla targetu, `prob` dla reszty
- Aktualizacja **LM Head** (wagi + biasy)
- Aktualizacja **EmbeddingTable** — gradient rozlozony rowno na wszystkie tokeny kontekstu (`inv_ctx = 1/ctx_size`)
- Aktualizacja wag **attention** ze zmniejszonym lr (`att_lr = lr * 0.01`)
- **Gradient clipping**: `math.clamp(..., -2.0, 2.0)` na wszystkich wagach

### Generowanie tekstu (`api.lua`)

- Buduje sekwencje embeddingów z aktualnych tokenow wejscia
- Przepuszcza przez attention → bierze ostatni wektor kontekstu
- Oblicza logity przez iloczyn skalarny z LM Head
- **Repetition penalty** — kara wykladnicza `penalty_factor ^ count` dla powtarzajacych sie tokenow
- Blokuje token `<PAD>` przez `-INF`
- Softmax z numeryczna stabilnoscia (odjecie `max_logit`)
- Weighted sampling przez kumulowane prawdopodobienstwo

---

## 🚀 Jak uruchomic

1. Otworz projekt w **Roblox Studio**
2. Upewnij sie ze struktura skryptow odpowiada drzewie modulu (`api`, `Tokenizer`, `Trainer`, `Transformer`, `data` jako rodzenstwo)
3. Odkomentuj linijke treningu w `api.lua`:
```lua
Trainer.Train(sentences_table, 600, 0.05, api.WygenerujTekstZKontekstem)
```
4. Uruchom — postep treningu bedzie widoczny w Output:
```
[Trainer] Inicjalizacja wag D=16...
[Trainer] Epoka 1/600 | Loss: 2.7081
[Trainer] Epoka 120/600 | Loss: 1.3204
[Trainer] Epoka 600/600 | Loss: 0.4912
[Trainer] Trening zakonczony!
```
5. Po treningu wywolaj generowanie:
```lua
local result = api.WygenerujTekstZKontekstem("kot pije", 5)
print(result)
```

---

## 🧪 Testowanie modelu

### Latwe — uzupelnianie par

| Prompt | Oczekiwany output |
|---|---|
| `kot pije` | `mleko` |
| `pies goni` | `kota` |
| `maly kot` | `spi` / `je` |
| `duzy pies` | `pilnuje` / `biega` |

### Srednie — relacje semantyczne

| Prompt | Oczekiwany output |
|---|---|
| `glodny kot` | `miauczy` / `je rybe` |
| `bialy pies` | `biega po ogrodzie` |
| `pies szczeka bo` | `widzi kota` |

### Trudne — dlugi kontekst

| Prompt | Oczekiwany output |
|---|---|
| `kot ucieka bo` | `boi sie psa` |
| `maly czarny` | `kot` |
| `stary pies spi` | `przy ogniu` |

---

## 📊 Parametry treningu

```lua
Trainer.Train(
    sentences_table,  -- ~20 zdan po tokenizacji
    600,              -- liczba epok
    0.05,             -- learning rate (LM Head + embeddingi)
                      -- attention lr = 0.05 * 0.01 = 0.0005
)
```

Trening nie blokuje glownego watku Roblox dzieki `task.wait()` po kazdej epoce.

---

## 💡 Wyzwania techniczne

- **Brak bibliotek ML w Lua** — caly backprop, softmax, gradient clipping napisane od zera
- **Numeryczna stabilnosc** — `math.clamp` przed kazdym `math.exp`, obsuga `NaN` przez `loss == loss`
- **Roblox Studio jako srodowisko ML** — `task.wait()` zapobiega zamrozeniu silnika podczas dlugiego treningu
- **Zarzadzanie pamiecia** — cache `ZERO_VECTOR`, lokalne referencje do tabel (`local vocab = Config.Vocab`) zamiast wielokrotnego indeksowania globalnego
- **Eksport wag** — `Config.exportConfigToOutput()` serializuje pelny stan modelu do konsoli Roblox po treningu

---

## 🛠 Technologie

- **Jezyk:** Lua 5.1 (Roblox)
- **Srodowisko:** Roblox Studio
- **Biblioteki zewnetrzne:** brak — 100% pure Lua

---

## 📚 Zrodla i inspiracje

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised) — Radford et al., 2018 (GPT-1)

---

---

# 🧠 LuaGPT-1: GPT-1 Implementation from Scratch in Roblox Studio <a name="english-version"></a>

> **EN** | [PL above](#)

---

## 📌 About the Project

LuaGPT-1 is a fully working GPT-1 architecture implementation written in **Lua**, built entirely from scratch inside **Roblox Studio** — with zero external ML libraries.

The project includes a complete pipeline: text tokenization, token embeddings, positional encoding, multi-layer causal self-attention, sigmoid feed-forward network, LM Head with softmax, and a manually implemented backpropagation algorithm.

---

## 🗂 Project Structure

```
LuaGPT-1/
├── api.lua          -- Entry point: training loop + text generation
├── Tokenizer.lua    -- Tokenization, vocab building, encode/decode
├── Transformer.lua  -- Self-Attention, Feed-Forward, Forward pass
├── Trainer.lua      -- Training loop, forward pass, backpropagation
└── data.lua         -- Config: embeddings, weights, vocab, layers
```

---

## ⚙️ Architecture

### Model Parameters

| Parameter | Value |
|---|---|
| Embedding dimension (D) | **16** |
| Attention layers | **12** |
| Attention heads | **12** |
| Context length | **4 tokens** |
| Vocabulary size | dynamic (built from training data) |
| FFN layers | **2** (16→32→16 with sigmoid) |
| Repetition penalty | **1.45** (exponential) |
| Temperature | **0.8** |

### Data Flow

```
Input (text)
      ↓
Tokenizer.Encode()         -- lowercase + cleanup + token ID mapping
      ↓
Token Embeddings           -- D=16 vector table per token
      ↓
[Self-Attention × 12]      -- Q/K/V projections + causal masking + residual
      ↓
Feed-Forward (sigmoid)     -- 16 → 32 → 16
      ↓
LM Head (dot product)      -- last_vector · weights[w_id] + bias
      ↓
Softmax + Temperature      -- probability normalization
      ↓
Weighted Sampling          -- sample from distribution
      ↓
Tokenizer.Decode()         -- ID → words
```

---

## 🔍 Implementation Details

### Tokenizer (`Tokenizer.lua`)

- Dynamically builds vocabulary from training text (`BuildVocab`)
- Preprocessing: lowercase, punctuation removal, dots treated as separate tokens
- Full `Encode` / `Decode` pipeline
- Zero-vector cache (`ZERO_VECTOR`) to avoid allocation on every `GetEmbedding` call
- Preserves trained weights if `EmbeddingTable` already exists

### Self-Attention (`Transformer.lua`)

- **Q, K, V, O** projections as per-dimension scalar vectors (simplified single-head implementation)
- **Causal masking** — token `i` only attends to tokens `j ≤ i`
- Scaled by `1/sqrt(D)`
- Numerical stability: `math.clamp(dot * scale, -20, 20)` before `math.exp`
- **Residual connection**: `seq[i] += attn_out[i]` after every layer
- 12 independent layers (`v1`..`v12`) with separate `Wq/Wk/Wv/Wo/Bq/Bk/Bv/Bo` weights

### Feed-Forward (`Transformer.lua`)

- 2 layers: `16 → 32 → 16`
- Activation function: **Sigmoid** (`1 / (1 + exp(-x))`)
- Weights initialized deterministically via `sin/cos`

### Backpropagation (`Trainer.lua`)

- Hand-written gradient through **cross-entropy loss**
- Logit gradient: `d_logit = prob - 1` for target token, `prob` for others
- Updates **LM Head** weights and biases
- Updates **EmbeddingTable** — gradient spread evenly across context tokens (`inv_ctx = 1/ctx_size`)
- Updates **attention weights** at reduced lr (`att_lr = lr * 0.01`)
- **Gradient clipping**: `math.clamp(..., -2.0, 2.0)` on all weights

### Text Generation (`api.lua`)

- Builds embedding sequence from current input tokens
- Passes through attention → takes last context vector
- Computes logits via dot product with LM Head weights
- **Repetition penalty** — exponential penalty `penalty_factor ^ count` for repeated tokens
- Blocks `<PAD>` token with `-INF`
- Numerically stable softmax (subtracts `max_logit`)
- Weighted sampling via cumulative probability

---

## 🚀 How to Run

1. Open the project in **Roblox Studio**
2. Make sure the module structure matches the tree (`api`, `Tokenizer`, `Trainer`, `Transformer`, `data` as siblings)
3. Uncomment the training line in `api.lua`:
```lua
Trainer.Train(sentences_table, 600, 0.05, api.WygenerujTekstZKontekstem)
```
4. Run — training progress will appear in Output:
```
[Trainer] Inicjalizacja wag D=16...
[Trainer] Epoka 1/600 | Loss: 2.7081
[Trainer] Epoka 120/600 | Loss: 1.3204
[Trainer] Epoka 600/600 | Loss: 0.4912
[Trainer] Trening zakonczony!
```
5. After training, call generation:
```lua
local result = api.WygenerujTekstZKontekstem("kot pije", 5)
print(result)
```

---

## 🧪 Testing the Model

### Easy — pair completion

| Prompt | Expected output |
|---|---|
| `kot pije` | `mleko` |
| `pies goni` | `kota` |
| `maly kot` | `spi` / `je` |
| `duzy pies` | `pilnuje` / `biega` |

### Medium — semantic relations

| Prompt | Expected output |
|---|---|
| `glodny kot` | `miauczy` / `je rybe` |
| `bialy pies` | `biega po ogrodzie` |
| `pies szczeka bo` | `widzi kota` |

### Hard — long context

| Prompt | Expected output |
|---|---|
| `kot ucieka bo` | `boi sie psa` |
| `maly czarny` | `kot` |
| `stary pies spi` | `przy ogniu` |

---

## 📊 Training Parameters

```lua
Trainer.Train(
    sentences_table,  -- ~20 sentences after tokenization
    600,              -- epochs
    0.05,             -- learning rate (LM Head + embeddings)
                      -- attention lr = 0.05 * 0.01 = 0.0005
)
```

Training does not block the main Roblox thread thanks to `task.wait()` after each epoch.

---

## 💡 Technical Challenges

- **No ML libraries in Lua** — backprop, softmax, and gradient clipping written entirely from scratch
- **Numerical stability** — `math.clamp` before every `math.exp`, NaN detection via `loss == loss`
- **Roblox Studio as an ML environment** — `task.wait()` prevents engine freeze during long training runs
- **Memory management** — `ZERO_VECTOR` cache, local table references (`local vocab = Config.Vocab`) to avoid repeated global indexing
- **Weight export** — `Config.exportConfigToOutput()` serializes the full model state to the Roblox Output console after training

---

## 🛠 Tech Stack

- **Language:** Lua 5.1 (Roblox)
- **Environment:** Roblox Studio
- **External Libraries:** none — 100% pure Lua

---

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017
- [Improving Language Understanding by Generative Pre-Training](https://openai.com/research/language-unsupervised) — Radford et al., 2018 (GPT-1)
