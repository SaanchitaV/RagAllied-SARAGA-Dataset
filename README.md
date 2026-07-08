# RagAllied-SARAGA-Dataset

# Swara-LM: A Self-Supervised Melodic Language Model for Carnatic Retrieval

Learning the *grammar of Carnatic melody* (sanchara) directly from audio, with no
raga labels — then using it to search Carnatic music at the phrase level.

---

## The one-paragraph idea

A language model learns English by predicting the next word in raw text; to win
that game it must internalize grammar and meaning, with nothing labeled. **This
project does the same for Carnatic music**: it turns recordings into sequences of
**swaras** (notes relative to the tonic) and trains a model to predict melodic
continuations. To succeed, the model must learn **sanchara** — the movement
grammar that actually defines a raga — *without ever being told the raga*. The
learned representations then power a **dense retrieval** system: embed any melodic
phrase, search for similar phrases, and evaluate whether the model separates
**allied ragas** (same notes, different movement) where naive note-matching fails.

---

## Why this design

The dataset (Saraga Carnatic, 197 recordings) is **small and long-tailed**: most
ragas have 1–3 recordings, too few to *supervise* a learned model per raga. But it
has **100% tonic + pitch coverage** — abundant raw melodic signal.

So the ML is moved **off** the scarce raga labels and **onto** the abundant pitch
data, via self-supervision. Raga labels are used only for a small **downstream
evaluation**, not for training. The small dataset stops being a blocker.

---

## Two components

### A — Swara Language Model  *(NLP / ML core; self-supervised)*
- Convert every recording → tonic-normalized **swara sequence** ("melodic sentence").
- Train a sequence model (Transformer / RNN) on **next-swara prediction** (and/or
  masked-swara modeling).
- **No labels needed.** The training signal is the melody itself.
- Output: contextual **phrase embeddings** that encode sanchara.

### B — Dense Melodic Retrieval  *(IR; uses A's embeddings)*
- Embed all phrases with A's learned representation.
- Index them in a **vector database** (FAISS / HNSW) for nearest-neighbor search.
- Query-by-phrase: given a melodic phrase, retrieve the most similar phrases.
- **Evaluation** (labels re-enter here, only for the 20 populated ragas):
  does a query phrase retrieve *same-raga* phrases and reject *allied-raga* ones?

---

## Pipeline stages

| Stage | What happens | Needs labels? |
|---|---|---|
| 0. Census | audit raga/artist/annotation depth | no |
| 1. Feature extraction | pitch → tonic-relative → swara sequences | no |
| 2. Tokenization | swara sequences → model tokens | no |
| 3. Pretrain (A) | self-supervised swara LM | **no** |
| 4. Embed + index (B) | phrase embeddings → vector index | no |
| 5. Retrieval | query-by-phrase nearest-neighbor search | no |
| 6. Evaluation | same-raga retrieval & allied-raga separation | yes (eval only) |

---

## Anti-cheating / validity measures

- **Tonic-relative** swara mapping — removes the tonic confound (uses Saraga's
  100% tonic annotations).
- **Per-recording normalization** — suppresses singer/mic signatures.
- **Held-out-artist evaluation** — split so that scoring well by memorizing a
  singer instead of learning grammar shows up as failure.
- **Baselines to beat:** (i) swara-presence bag-of-notes vector, (ii) swara
  **transition** (Markov) vector. The learned LM must beat both on allied-raga
  retrieval, or the negative result is itself reported.

---

## Evaluation

- **Task:** phrase → retrieve same-raga phrases (query-by-example).
- **Hard cases:** allied ragas present in the data (e.g. the Saveri / Gaula /
  Ritigaula cluster), reported **per category** (scale-identical, same-janaka,
  gamaka-only).
- **Metrics:** Precision@k, MAP, MRR (nDCG if graded relevance defined).
- **Comparison:** learned LM vs. transition floor vs. bag-of-notes baseline.

---

## Tech stack

| Layer | Tools |
|---|---|
| Data | Saraga Carnatic (Kaggle mirror); Dunya/CompMusic for scale-up |
| Loading / features | file-parsing + `compiam` (tonic, pitch) |
| Sequence modeling | PyTorch (Transformer / RNN) |
| Retrieval | FAISS or hnswlib (dense ANN) |
| Analysis | pandas, numpy |

---

## Roadmap

- [x] Census — inventory, depth, annotation coverage
- [x] Feature extraction — swara sequences from all 197 recordings
- [x] Baselines — bag-of-notes + transition vectors
- [ ] Swara LM (A) — self-supervised pretraining
- [ ] Dense retrieval (B) — embed, index, query
- [ ] Evaluation — same-raga retrieval, allied-raga separation, per category
- [ ] (Stretch) Scale up with Dunya data; fine-tune for allied-raga discrimination

---

