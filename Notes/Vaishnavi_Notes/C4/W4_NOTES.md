# COURSE 4
## WEEK 4 - Transformers

## Why Transformers?

- RNNs process sequences step-by-step → can't parallelize, slow to train
- Attention mechanism alone is enough — no recurrence needed
- "Attention is All You Need" (Vaswani et al., 2017) — this paper changed everything
- Transformers can attend to all positions simultaneously → much faster training

## Self-Attention

- Each token attends to every other token in the sequence
- Three vectors for each token: **Query (Q)**, **Key (K)**, **Value (V)**
- Q, K, V = linear projections of the input embedding
- Attention score: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
- The sqrt(d_k) scaling prevents dot products from getting too large

## Multi-Head Attention

- Run self-attention h times in parallel with different learned projections
- Concatenate outputs → linear projection
- Different heads can learn different types of relationships
  (e.g., one head for syntax, another for semantics)
- In the paper: h = 8 heads, d_k = 64

## Positional Encoding

- Transformers have no built-in notion of order (unlike RNNs)
- Add positional encoding to input embeddings to inject position info
- Sinusoidal encoding: PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- Can also use learned positional embeddings

## Transformer Architecture (Encoder-Decoder)

Encoder:
  - N identical layers (typically N=6)
  - Each layer: Multi-Head Self-Attention → Add & Norm → Feed Forward → Add & Norm
  - Processes the source sequence

Decoder:
  - N identical layers
  - Each layer: Masked Multi-Head Self-Attention → Add & Norm
                → Cross-Attention with encoder output → Add & Norm
                → Feed Forward → Add & Norm
  - Masked attention prevents decoder from "seeing" future tokens (autoregressive)

## BERT and GPT:

- **BERT**: encoder-only transformer, trained with masked language modeling
  → great for understanding tasks (classification, NER, QA)
- **GPT**: decoder-only transformer, trained with language modeling (next token prediction)
  → great for generation tasks

## Transformers in our project:

- ASR Transformer: encoder processes audio features (MFCCs/spectrograms), decoder outputs text
- NMT Transformer: standard encoder-decoder, source language → English
- Key difference from BERT/GPT: both use full encoder-decoder architecture
- Training on Bhaashaanuvad showed transformer ASR outperforms BiLSTM+CTC
  (WER: 14.7% vs 18.4%)
