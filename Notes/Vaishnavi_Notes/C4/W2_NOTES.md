# COURSE 4
## WEEK 2 - LSTMs & GRUs

## GRU (Gated Recurrent Unit)

- Simplified version of LSTM, works surprisingly well in practice
- Two gates: **update gate** (Γu) and **reset gate** (Γr)
- Memory cell c<t> = Γu * c_tilde<t> + (1 - Γu) * c<t-1>
- c_tilde<t> = tanh(Wc[Γr * c<t-1>, x<t>] + bc)
- When Γu is close to 0, the cell c<t> ≈ c<t-1> (memory is preserved across many timesteps)
- This solves the vanishing gradient problem for long sequences

## LSTM (Long Short-Term Memory)

- More powerful than GRU but also more computationally expensive
- Has 3 gates:
  - **Forget gate** Γf: decides what to forget from previous cell state
  - **Update gate** Γu: decides what new info to store
  - **Output gate** Γo: decides what to output from cell state
- Separate cell state c<t> and hidden state a<t>
- Cell update: c<t> = Γu * c_tilde<t> + Γf * c<t-1>
- Output: a<t> = Γo * tanh(c<t>)

## When to use GRU vs LSTM?

- GRU: simpler, faster to train, works well for shorter sequences
- LSTM: better for very long-range dependencies, more expressive
- In practice: try both and see — for this project we went with BiLSTM for the CTC model

## Bidirectional RNNs (BRNN)

- Standard RNN only looks at past context — sometimes future context helps too
- BRNN runs one forward RNN + one backward RNN, concatenates their hidden states
- Very useful for ASR because the model can "hear" the whole word before deciding
- Downside: need the entire sequence before making any prediction (not suitable for true real-time)

## Deep RNNs

- Stack multiple RNN layers on top of each other
- Each layer's output becomes the input to the next
- Usually 2-3 layers is enough for RNNs (unlike CNNs where you can go much deeper)
- Notation: a[l]<t> = activation at layer l, timestep t
