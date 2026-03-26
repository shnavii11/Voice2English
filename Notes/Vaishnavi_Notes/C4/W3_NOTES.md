# COURSE 4
## WEEK 3 - Sequence to Sequence & Attention Mechanism

## Seq2Seq Models

- Two RNNs: encoder reads input sequence → produces a context vector
- Decoder takes context vector → generates output sequence
- Works well for machine translation, summarization, image captioning
- Problem: the entire input is compressed into a single fixed-size vector
  → bottleneck for long sequences, the encoder "forgets" early parts

## Attention Mechanism

- Instead of one context vector, attention lets the decoder "look at" all encoder hidden states
- At each decoder step, attention computes a weighted sum of encoder outputs
- Weights (α) tell the decoder which parts of the input to focus on
- α<t,t'> = attention weight for output t, input t'
- These weights are computed using a small neural network (alignment model)
- Context vector: c<t> = Σ α<t,t'> * a<t'>

## Why attention works so well:

1. Decoder can access any part of the input at any time
2. The alignment weights are interpretable — you can visualize what the model is "attending to"
3. Works much better for long sequences compared to vanilla seq2seq

## Attention in our NMT model:

- Encoder: BiLSTM over the ASR transcript (source language)
- Decoder: LSTM with attention over encoder outputs
- At each decoder timestep → compute attention weights over all source tokens
- Significantly improved translation quality for longer sentences

## Speech Attention (for ASR):

- In CTC-based ASR, we don't use attention directly
- In Transformer ASR, self-attention replaces recurrence
- Cross-attention between encoder speech features and decoder text tokens is key
  to good alignment between audio frames and text tokens

## Notes:

- Attention complexity: O(n*m) where n = source length, m = target length
- For long audio sequences this can be expensive → use local attention or windowed attention
- The alignment visualization (heatmap of α values) is a great debugging tool

## Beam Search (Decoding)

- Greedy decoding picks the highest probability token at each step — can miss better overall sequences
- Beam search keeps top B candidates (beam width) at each step
- Typically B=5 or B=10 — larger B improves quality but increases compute
- Length normalization: divide log-probability by sequence length to avoid bias toward short outputs
  → normalized score = (1/Ty) * Σ log P(y<t> | x, y<1..t-1>)

## Bleu Score (recap)

- Measures overlap between machine output and reference(s)
- Precision based: how many n-grams in the hypothesis appear in the reference
- Modified precision to avoid rewarding repetition
- Combined score: BLEU = BP * exp(Σ wn * log pn)
  where BP = brevity penalty, pn = n-gram precision
