# COURSE 4
## WEEK 1 - Recurrent Neural Networks

- Standard neural nets don't work well for sequential data like audio, text, video because input/output sizes vary and they don't share features across positions.
- RNNs process sequences step by step, passing hidden state from one timestep to the next — this is what gives them "memory".
- At each timestep t, the hidden state is: a<t> = g(Waa * a<t-1> + Wax * x<t> + ba)
- The output at each step: y_hat<t> = g(Wya * a<t> + by)
- Problem with vanilla RNNs — they struggle to learn long-range dependencies (vanishing gradients)

## Types of RNN architectures:

1. **Many to Many** — input and output both are sequences (e.g., machine translation)
2. **Many to One** — sequence input, single output (e.g., sentiment classification)
3. **One to Many** — single input, sequence output (e.g., music generation)
4. **One to One** — standard neural net

## Language Modelling with RNNs

- Given a sentence, predict probability of the next word
- Training: tokenize text → build vocab → feed tokens one at a time → compute loss using softmax
- At test time: sample from the predicted distribution to generate new text

## Vanishing & Exploding Gradients

- Gradients can shrink exponentially through many layers → model can't learn long-range dependencies
- Exploding gradients: use gradient clipping (cap the gradient norm)
- Vanishing gradients: harder to solve → leads to GRU and LSTM (next week)

## Gated Recurrent Unit (GRU) — preview

- Introduces a "memory cell" c<t> and update/reset gates
- Update gate Γu ∈ [0,1] decides how much of the old memory to keep
- When Γu ≈ 0, the cell c<t> ≈ c<t-1> → remembers information from many steps back
