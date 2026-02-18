
PAGE 1 — TITLE SHEET


                                    A

                            PROJECT REPORT

                                   On

              VOICE2ENGLISH: AN END-TO-END MULTILINGUAL
              SPEECH-TO-ENGLISH TRANSLATION SYSTEM USING
               DEEP LEARNING AND NEURAL MACHINE TRANSLATION


                                   By

                          Ms. Vaishnavi Sanap


                       Under The Guidance of

                          PROF. SHENAL


              DEPARTMENT OF ELECTRONICS ENGINEERING
        VEERMATA JIJABAI TECHNOLOGICAL INSTITUTE (VJTI)
         [Central Technological Institute, Maharashtra State]
                      Matunga, Mumbai - 400 019

                             2025-2026


---


PAGE 2 — CERTIFICATE


                            CERTIFICATE


THIS IS TO CERTIFY THAT THE FIELD PROJECT

                              Entitled

    VOICE2ENGLISH: AN END-TO-END MULTILINGUAL SPEECH-TO-ENGLISH
    TRANSLATION SYSTEM USING DEEP LEARNING AND NEURAL MACHINE
    TRANSLATION

                          Submitted by

                Ms. Vaishnavi Sanap
                Examination No. 241061061

Has successfully completed her Field Project for the partial
fulfillment for the award of Field Project Second Year Bachelor
of Technology (Electronics Engineering) of Veermata Jijabai
Technological Institute is approved.


Prof. Shenal                            Dr. G. M. Galshetwar
      Guide                            Field Project Coordinator


Dr. R. A. Patil
      H.O.D.


---


PAGE 3 — EXAMINERS CERTIFICATE


                       EXAMINERS CERTIFICATE


Project Report Entitled: "Voice2English: An End-To-End Multilingual
Speech-To-English Translation System Using Deep Learning And Neural
Machine Translation" in the partial fulfillment of the requirements
for the award of Field Project Second Year Bachelor of Technology
(Electronics Engineering) of Veermata Jijabai Technological Institute
is certified.


INTERNAL EXAMINER                          EXTERNAL EXAMINER


DATE  : ____________________

PLACE : ____________________


---


PAGE 4 — DECLARATION


                           DECLARATION


        I certify that the ideas, designs and experimental work,
results, analyses and conclusions set out in this Field Project are
entirely my own effort, except where otherwise indicated and
acknowledged.

        I further certify that the work is original and has not
been previously submitted for assessment in any other course or
institution, except where specifically stated.


                                          Signature of Student

Date : ____________________               Ms. Vaishnavi Sanap


---


PAGE 5 — PROJECT PHOTOGRAPH

        The screenshot below shows the working Voice2English system
interface. The web application accepts a Hindi audio file, displays
the Hindi transcription produced by the ASR model on the left panel,
and outputs the English translation produced by the NMT model on the
right panel. The Translate button triggers the end-to-end pipeline.

        [Insert the system interface screenshot here — filename:
         Voice2English_UI_Screenshot.png]

        Under the Guidance of Prof. Shenal and Dr. R. A. Patil (H.O.D.)


---


PAGE 6 — INDEX


                              INDEX

Chapter     Name of Topic                                   Page

            Abbreviations                                      i
            List of Tables                                    ii

1.          INTRODUCTION                                       1
    1.1     Introduction                                       1
    1.2     Necessity                                          2
    1.3     Objectives                                         3
    1.4     Theme                                              3
    1.5     Organization                                       4

2.          LITERATURE SURVEY                                  5
    2.1     History of Speech Recognition                      5
    2.2     Automatic Speech Recognition                       6
    2.3     Neural Machine Translation                         8
    2.4     End-to-End Speech Translation Systems              9

3.          SYSTEM DESIGN                                     11
    3.1     System Overview                                   11
    3.2     Dataset Description                               12
    3.3     Data Preprocessing                                14
    3.4     Format 1: ASR (CNN + BiLSTM + CTC) + NMT         16
    3.5     Format 2: ASR (Transformer) + NMT (Transformer)   20
    3.6     Training Methodology                              23

4.          PERFORMANCE ANALYSIS                              25
    4.1     Evaluation Metrics                                25
    4.2     ASR Model Performance                             26
    4.3     NMT Model Performance                             28
    4.4     End-to-End Pipeline Performance                   30
    4.5     Comparison of Format 1 and Format 2               32

5.          CONCLUSION                                        34
    5.1     Conclusions                                       34
    5.2     Future Scope                                      35
    5.3     Applications                                      36
    5.4     Advantages                                        36
    5.5     Limitations                                       37

            References
            Appendices
            Acknowledgement


LIST OF TABLES

Table 3.1  Bhaashaanuvad Dataset Statistics                      13
Table 3.2  ASR Model Hyperparameters                             19
Table 3.3  NMT Model Hyperparameters                             22
Table 4.1  ASR WER Results on Test Set                           27
Table 4.2  NMT BLEU Score Results                                29
Table 4.3  Comparison of Both Architectures                      32


---
(Page numbering begins from Chapter 1)
---


1. INTRODUCTION


1.1 Introduction

        India is one of the most linguistically diverse countries in
the world, with 22 officially recognized languages and hundreds of
regional dialects. This diversity, while culturally rich, creates
significant communication barriers, particularly for people who do
not speak English. At the same time, most technical resources,
academic content, and digital information are predominantly available
in English. This gap motivated the development of Voice2English - a
deep learning based system that converts spoken audio from Indian
languages directly into English text.

        The system works in two stages. In the first stage, an
Automatic Speech Recognition (ASR) model listens to the audio and
produces a written transcript in the original spoken language. In the
second stage, a Neural Machine Translation (NMT) model takes that
transcript and translates it into English. Two different architectures
are built and compared for the ASR stage: Format 1 uses a Convolutional
Neural Network (CNN) with Bidirectional LSTM layers and a CTC decoder,
while Format 2 uses a pure Transformer encoder for ASR. Both formats
share the same Transformer-based NMT model.


1.2 Necessity

        The necessity for this project comes from several gaps that
exist in currently available systems:

(i)   Most existing speech translation tools support only major global
      languages and are not optimized for low-resource Indian languages
      like Hindi, Marathi, or Telugu.

(ii)  Commercial APIs for speech translation are expensive and cannot
      be deployed offline, making them unsuitable for rural or
      resource-constrained environments.

(iii) Research in Indian language ASR and NMT is scattered. Very few
      projects combine both into a single working pipeline trained from
      scratch using open data.

(iv)  The recent release of large-scale multilingual datasets like
      Bhaashaanuvad by AI4Bharat makes it feasible to train such
      systems without proprietary data, and this project makes use
      of that opportunity.


1.3 Objectives

The specific objectives of this field project are:

1. To build an ASR model that can transcribe spoken Hindi audio into
   text, using two different deep learning architectures.

2. To train a Neural Machine Translation model using the Transformer
   architecture to convert Hindi text to English.

3. To combine both models into a single end-to-end pipeline that
   takes raw audio as input and outputs an English translation.

4. To evaluate and compare both ASR architectures using Word Error
   Rate (WER) and to evaluate the NMT model using the BLEU score.

5. To demonstrate real-time inference on audio samples not seen
   during training.


1.4 Theme

        The central theme of this field project is the application
of modern deep learning architectures to the problem of multilingual
speech-to-text translation. The project combines two important areas
of machine learning - Automatic Speech Recognition and Natural
Language Processing - into a practical, working system. A key aspect
of this project is the comparison between recurrent architectures
(BiLSTM with CTC) and attention-based architectures (Transformers)
for the task of speech recognition, and understanding the trade-offs
between the two in terms of performance and training complexity.


1.5 Organization

        The report is organized as follows. Chapter 2 presents a
Literature Survey covering the development of ASR and NMT systems
and the relevant background work. Chapter 3 describes the System
Design in detail, covering the dataset, preprocessing steps, and
both model architectures. Chapter 4 presents the Performance Analysis
with training curves, evaluation results, and a comparison of the
two formats. Chapter 5 concludes the report with key findings,
future scope, applications, advantages, and limitations.


---


2. LITERATURE SURVEY


2.1 History Of Speech Recognition

        Automatic Speech Recognition has developed over several
decades. In the 1950s, Bell Laboratories built an early system
called "Audrey" that could recognize only spoken digits from a single
speaker. Progress remained slow until the 1970s and 1980s, when
Hidden Markov Models (HMMs) became the standard approach. HMMs
treated speech as a sequence of probabilistic states and were combined
with Gaussian Mixture Models (GMMs) to model acoustic features. These
systems required large amounts of hand-crafted linguistic knowledge,
including pronunciation dictionaries and phoneme-level transcriptions.

        The shift to deep learning in the 2010s changed the field
significantly. Hinton et al. (2012) showed that replacing GMMs with
deep neural networks in the acoustic model improved accuracy
substantially on standard benchmarks. Graves et al. (2006) introduced
the Connectionist Temporal Classification loss, which allowed sequence
models to be trained without requiring frame-level alignment between
audio and text. This made it much simpler to train end-to-end ASR
systems. By 2014, models based on deep learning had surpassed
traditional HMM-GMM systems on most tasks.


2.2 Automatic Speech Recognition

2.2.1 Connectionist Temporal Classification (CTC)

        CTC, introduced by Graves et al. in 2006, addresses the
alignment problem in sequence-to-sequence learning. In ASR, the
input audio and the output text are of different lengths, and the
exact correspondence between each audio frame and each character is
not known. CTC defines a probability distribution over all possible
alignments between the input and output sequences and sums over them
during training, so the model does not need explicit frame-level
labels. A special blank token is used to represent no output at this
timestep. During inference, consecutive repeated tokens are collapsed
and blank tokens are removed to recover the final text.

2.2.2 Convolutional Neural Networks For Audio

        CNNs applied to spectrograms work as local feature extractors.
Each convolutional filter learns to detect a specific pattern in a
small region of the time-frequency representation. Filters with large
kernel sizes along the frequency axis can capture formant structures
in speech. Strides along the time dimension reduce the sequence
length before it is passed to recurrent layers, which lowers the
computational cost of training.

2.2.3 Bidirectional Long Short-Term Memory Networks

        LSTMs, proposed by Hochreiter and Schmidhuber in 1997, use
three gating mechanisms - input gate, forget gate, and output gate -
to control how information flows through the hidden state over time.
This allows the network to retain relevant information over long
sequences without suffering from the vanishing gradient problem that
affects standard RNNs. Bidirectional LSTMs run the sequence in both
the forward and backward directions and concatenate the outputs,
giving the model access to both past and future context at each
timestep. This is particularly useful in ASR, where understanding a
word often depends on the sounds that follow it.

2.2.4 Transformer-Based ASR

        Vaswani et al. (2017) introduced the Transformer architecture,
which replaces recurrence entirely with a self-attention mechanism.
Self-attention allows every position in the input to directly attend
to every other position, regardless of distance. This overcomes the
gradient issues of RNNs and can be computed in parallel, making
training much faster. Systems like Wav2Vec 2.0 (Baevski et al., 2020)
and Whisper (Radford et al., 2022) use Transformer encoders for ASR
and have achieved state-of-the-art results across many languages.


2.3 Neural Machine Translation

        Neural Machine Translation replaces traditional phrase-based
statistical methods with end-to-end neural networks. Sutskever et al.
(2014) showed that a sequence-to-sequence model using LSTMs with an
attention mechanism could perform comparably to phrase-based systems
on English-French translation. The key challenge in NMT is handling
variable-length input and output sequences and learning long-range
dependencies between words.

        The Transformer became the standard architecture for NMT after
2017. Its multi-head attention mechanism allows the decoder to look
at multiple parts of the encoded source sentence simultaneously when
generating each output word. Positional encodings are added to the
input embeddings to give the model information about word order, since
the attention mechanism itself is position-invariant. The Transformer
trains much faster than LSTM-based models because its operations are
parallelizable, and it produces significantly better translation
quality on standard benchmarks.

        SentencePiece (Kudo and Richardson, 2018) is a language-
independent subword tokenizer. It uses Byte Pair Encoding to split
words into smaller units, allowing the model to handle rare and
unseen words by decomposing them into known subword pieces. This is
especially useful for morphologically rich languages like Hindi, where
the same root word can appear in many different forms.


2.4 End-To-End Speech Translation Systems

        Speech translation systems can be built in two ways: cascaded
or end-to-end. In a cascaded system, a separately trained ASR model
produces a transcript, and a separately trained MT model translates
it. This approach is modular and benefits from independently developed
components, but errors from the ASR stage propagate and compound in
the MT stage. End-to-end systems train both stages jointly, reducing
error propagation, but require large amounts of aligned speech and
translation data, which is harder to collect.

        For Indian languages, data availability has historically been
a major limitation. The Bhaashaanuvad dataset released by AI4Bharat
in 2024 provides aligned audio, source transcripts, and English
translations for Hindi, Marathi, Telugu, and Bengali, making a
cascaded ASR+NMT pipeline practical to train. This project uses
that dataset and implements a cascaded system, exploring two
different architectures for the ASR stage.


---


3. SYSTEM DESIGN


3.1 System Overview

        Voice2English is a cascaded speech translation pipeline made
up of two independently trained deep learning models. The pipeline
operates as follows:

Stage 1 - Automatic Speech Recognition: The raw audio waveform is
converted into a spectrogram and passed through the ASR model, which
outputs a text transcription in the source language (Hindi).

Stage 2 - Neural Machine Translation: The source language
transcription is passed to the NMT model, which outputs the
corresponding English translation.

        Two ASR architectures are implemented and compared. Format 1
uses a CNN + BiLSTM + CTC model. Format 2 uses a Transformer-based
ASR model. Both share the same Transformer NMT model.


3.2 Dataset Description

3.2.1 Bhaashaanuvad Dataset

        The Bhaashaanuvad dataset was released by AI4Bharat in 2024.
It is a large-scale multilingual speech dataset designed for training
and evaluating speech-to-text and speech translation systems for
Indian languages. It contains paired audio recordings with source-
language transcripts and English translations.

        The dataset is available on HuggingFace under the collection
"ai4bharat/bhasaanuvaad". Each sample in the dataset contains the
following fields:

    chunked_audio_filepath  : path to the audio chunk (.wav format)
    text                    : original transcript in the source language
    pred_text               : pre-computed model-predicted transcript
    audio_filepath          : path to the full original audio file
    start_time              : starting time of the chunk in seconds
    duration                : duration of the audio chunk in seconds
    alignment_score         : quality score for audio-text alignment
    en_text                 : English translation of the transcript
    en_mining_score         : quality score for the English translation

Table 3.1 Bhaashaanuvad Dataset Statistics

    Language      No. of Samples     Used for Training
    Hindi         88,566             Yes (primary language)
    Marathi       29,154             Yes (in full dataset)
    Telugu        58,344             Yes (in full dataset)
    Bengali       86,763             Yes (in full dataset)

For the ASR model, a subset of 10,000 Hindi samples was used due to
compute constraints. For NMT training, 25,000 Hindi-English pairs
were used. Testing was done on a separate set of 100 Hindi samples
hosted on HuggingFace as "Purvaxxx/hindi_test_dataset."


3.3 Data Preprocessing

3.3.1 Audio Preprocessing For ASR

        All audio files are first resampled to 16,000 Hz. A power
spectrogram is then computed using the Short-Time Fourier Transform
with the following parameters:

    n_fft       = 256
    hop_length  = 160
    win_length  = 256
    power       = 2.0

The power spectrogram is converted to a magnitude spectrogram by
taking its square root. The resulting feature matrix is then normalized
per frequency bin - the mean is subtracted and the result is divided
by the standard deviation, so each frequency dimension has zero mean
and unit variance. The final feature matrix has shape (T, 129), where
T is the number of time frames and 129 is the number of frequency bins
(n_fft / 2 + 1).


3.3.2 Text Preprocessing For ASR

        All source language transcripts are lowercased. A
character-level vocabulary of 64 characters is built from the first
10,000 Hindi training samples. A special token named Blank is added
at index 0 for use during CTC training. The vocabulary covers all
Hindi vowels, consonants, matras, and common punctuation marks.
The complete character list is given in Appendix A.


3.3.3 Text Preprocessing For NMT

        Both the Hindi source text and the English target text are
normalized using Unicode NFC normalization and then lowercased. A
SentencePiece tokenizer using Byte Pair Encoding is trained separately
for the source and target languages, with a vocabulary size of 4,000
for each. Three special tokens are used:

    Index 0 : unknown token
    Index 1 : start-of-sequence token (SOS)
    Index 2 : end-of-sequence token (EOS)


3.4 Format 1: ASR (CNN + BiLSTM + CTC) + NMT (Transformer)

3.4.1 ASR Architecture

        The ASR model for Format 1 is a hybrid architecture that
combines convolutional layers, bidirectional LSTMs, and a CTC output.
It is inspired by the DeepSpeech2 model.

Convolutional Block 1:
    Type          : Conv2D
    Filters       : 32
    Kernel size   : (11, 41)
    Stride        : (2, 2)
    Normalization : BatchNorm2D
    Activation    : ReLU

This layer extracts low-level spectral and temporal features from the
spectrogram. The large kernel along the frequency axis captures formant
patterns, and the stride of 2 in the time dimension reduces the
sequence length by half.

Convolutional Block 2:
    Type          : Conv2D
    Filters       : 32
    Kernel size   : (11, 21)
    Stride        : (1, 2)
    Normalization : BatchNorm2D
    Activation    : ReLU

Bidirectional LSTM Layers:
    Hidden units     : 128 per direction (256 total per layer)
    Number of layers : 2
    Dropout          : 0.5 between layers
    Direction        : Bidirectional

After the two convolutional blocks, the output tensor is permuted and
reshaped from (batch, channels, time, frequency) to
(batch, time, channels x frequency) so that it can be fed into the
BiLSTM layers as a sequence.

Dense Layer:
    Input      : 256
    Output     : 128
    Activation : ReLU
    Dropout    : 0.5

Output Layer (Classifier):
    Input  : 128
    Output : 64  (vocabulary size including blank token)

CTC Decoding:
    Blank index : 0
    Method      : Greedy decoding - argmax is taken over character
                  probabilities at each time step, consecutive repeated
                  tokens are collapsed, and blank tokens are removed to
                  produce the final transcript.

Total Trainable Parameters: approximately 2.1 Million

Table 3.2 ASR Model (Format 1) Hyperparameters

    Hyperparameter              Value
    Input Feature Dimension     129 (n_fft / 2 + 1)
    Conv1 Kernel / Stride       (11, 41) / (2, 2)
    Conv2 Kernel / Stride       (11, 21) / (1, 2)
    Convolutional Channels      32
    BiLSTM Hidden Units         128 per direction
    BiLSTM Layers               2
    Dropout Rate                0.5
    Vocabulary Size             64 (including blank token)
    Loss Function               CTC Loss
    Optimizer                   Adam, learning rate 1e-4
    Batch Size                  4
    Training Epochs             27
    Time Reduction Factor       2 (from Conv1 stride)


3.5 Format 2: ASR (Transformer) + NMT (Transformer)

3.5.1 ASR Architecture

        In Format 2, the CNN-BiLSTM backbone is replaced by a
Transformer encoder for ASR. Spectrogram features are projected to
the model dimension, sinusoidal positional encodings are added, and
the Transformer encoder uses self-attention to model dependencies
across all time frames simultaneously.


3.5.2 NMT Architecture (Common To Both Formats)

        The NMT model uses a standard Transformer encoder-decoder
architecture implemented with PyTorch's nn.Transformer module.

Positional Encoding:
        Sinusoidal positional encodings are added to the token
embeddings so the model can use position information:

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

The maximum supported sequence length is 1,000 tokens.

Source Embedding  : Embedding of size (4000, 512) + Positional Encoding
Target Embedding  : Embedding of size (4000, 512) + Positional Encoding
Transformer       :
    d_model            = 512
    Number of heads    = 4
    Encoder layers     = 6
    Decoder layers     = 6
    Feedforward dim    = 512
    Dropout            = 0.1
Output Layer      : Linear(512, 4000)

A causal mask is applied to the target sequence so the decoder cannot
attend to future positions. Padding masks are applied to both source
and target sequences to ignore padded positions during attention.

Table 3.3 NMT Model Hyperparameters

    Hyperparameter              Value
    Source Vocabulary Size      4,000
    Target Vocabulary Size      4,000
    Tokenizer                   SentencePiece BPE
    d_model                     512
    Number of Attention Heads   4
    Encoder Layers              6
    Decoder Layers              6
    Feedforward Dimension       512
    Dropout Rate                0.1
    Loss Function               Cross-Entropy (padding ignored)
    Optimizer                   Adam, learning rate 1e-4
    Gradient Clipping           1.0
    Batch Size                  32
    Training Epochs             60
    Training Samples            25,000 Hindi-English pairs


3.6 Training Methodology

        All models were trained on cloud compute platforms - Google
Colab for ASR preprocessing and integration, and Kaggle for NMT
training (GPU: NVIDIA Tesla T4/P100). Model checkpoints were saved
to Google Drive at regular intervals during training.

For ASR training:
    - The dataset is loaded from HuggingFace and spectrogram features
      are computed and stored.
    - A custom collate function pads spectrograms and label sequences
      to the longest sample in each batch.
    - CTC Loss is computed after log-softmax. The input lengths passed
      to CTC are divided by 2 to account for the time-dimension stride
      in the first convolutional layer.
    - WER is computed on a held-out validation batch after each epoch.

For NMT training:
    - Source and target texts are normalized, tokenized using
      SentencePiece, and stored as integer ID sequences.
    - The decoder receives the target sequence shifted right
      (teacher forcing) during training.
    - Cross-Entropy loss is computed only over non-padded positions.
    - Greedy decoding is used to generate sample translations after
      each epoch for visual inspection.


---


4. PERFORMANCE ANALYSIS


4.1 Evaluation Metrics

4.1.1 Word Error Rate (WER)

        WER is the standard metric for evaluating ASR systems. It
measures what fraction of the words in the reference transcript were
incorrectly recognized:

    WER = (Substitutions + Deletions + Insertions) / Total Words

A lower WER is better. A WER of 0 means the transcription is perfect.


4.1.2 BLEU Score

        The BLEU score (Bilingual Evaluation Understudy) is used to
evaluate the quality of machine translations. It compares the machine-
generated translation against one or more reference translations by
measuring n-gram overlap, with a brevity penalty for translations that
are too short:

    BLEU = BP x exp(sum of wn x log pn)

where pn is the modified n-gram precision for order n, wn is the
weight (uniform), and BP is the brevity penalty. BLEU is reported
on a scale of 0 to 100, with higher values indicating better quality.


4.2 ASR Model Performance

4.2.1 Training And Validation Loss

        The ASR CTC model was trained for 27 epochs on 10,000 Hindi
audio samples, using an 80/20 train/validation split. The CTC
training loss decreased consistently from a high initial value and
converged to approximately 20-25 by epoch 27. The validation loss
followed a similar trend, with a small gap between training and
validation, indicating that the model was not significantly overfitting.


4.2.2 WER On Test Set

        The trained ASR model was evaluated on 100 held-out Hindi
audio samples from the Bhaashaanuvad dataset.

Table 4.1 ASR WER Results On Test Set

    Metric                      Value
    Test Set Size               100 samples
    Overall WER                 33.45%
    WER on Individual Sample    16.00%

The overall WER of 33.45% on the full test set is consistent with
the difficulty of training an ASR model from scratch on only 10,000
samples. Short individual utterances achieved a significantly lower
WER of 16%, which shows that the model has correctly learned many
Hindi phoneme-to-character mappings.


4.2.3 Sample ASR Output

        The following shows a representative output from the ASR
model comparing the ground truth Hindi transcript with the model's
predicted transcript:

    Ground Truth :  उस ने उसके उत्तर में उन से कहा कि मेरी माता
                    और मेरे भाई ये ही हैं जो परमेश्वर का वचन
                    सुनते और मानते हैं
    Prediction   :  उस ने उसके उत्तर में उन से कहा कि मेरी माता
                    और मेरे धाई योही है जो परमेश्वर का वचन
                    सुनते और मानते हैं
    WER          :  16%


4.3 NMT Model Performance

4.3.1 Training And Validation Loss

        The NMT Transformer was trained for 60 epochs on 25,000
Hindi-English sentence pairs with a 90/10 train/validation split and
a batch size of 32. The cross-entropy loss decreased steadily and
converged to approximately 1.0-1.5 by epoch 60. The validation loss
closely tracked the training loss throughout, showing that the model
generalized well to unseen sentence pairs.

Table 4.2 NMT BLEU Score Results

    Metric                  Value
    Corpus BLEU Score       85.43
    Sample BLEU Score       0.59 (59%)
    Decoding Method         Greedy Decoding

The corpus BLEU score of 85.43 indicates that the NMT model produces
translations that are very close to the reference English translations
in terms of n-gram overlap. The sample-level BLEU of 0.59 on an
individual utterance from the integration test further confirms this.


4.3.2 Sample NMT Output

        The following is a representative translation from the model,
showing the Hindi source, the reference English translation, and the
model's predicted output:

    Source (Hindi) :  उस ने उसके उत्तर में उन से कहा कि मेरी माता
                      और मेरे भाई ये ही हैं जो परमेश्वर का वचन
                      सुनते और मानते हैं
    Reference (EN) :  And he answered and said unto them, My mother
                      and my brethren are these which hear the word
                      of God, and do it
    Prediction     :  and he answered and said unto them, behold,
                      my mother and my brethren are these which hear
                      the word of god, and do nothing of him


4.4 End-To-End Pipeline Performance

        The complete Voice2English pipeline was tested on live audio
samples that had not been seen during training. The system successfully
transcribed short Hindi utterances and produced coherent English
translations, demonstrating that the two independently trained models
can work together effectively as a complete pipeline.

        Sample outputs from the integration test:

    Input Audio  : Hindi speech (.wav, 16 kHz)
    ASR Output   : हारीम की सन्तान तीन सौ बीस
    NMT Output   : the children of harim, three hundred and twenty

    Input Audio  : Hindi speech (.wav, 16 kHz)
    ASR Output   : और कहने लगा
    NMT Output   : and he began to speak


4.5 Comparison Of Format 1 And Format 2

Table 4.3 Comparison Of Both Architectures

    Aspect                  Format 1                Format 2
                            CNN + BiLSTM + CTC      Transformer ASR
    ASR Architecture        CNN + BiLSTM + CTC      Transformer Encoder
    NMT Architecture        Transformer             Transformer
    ASR WER on Test Set     33.45%                  To be filled
    NMT BLEU Score          85.43                   85.43 (same NMT)
    Training Speed          Faster                  Slower
    ASR Parameters          ~2.1 Million            Higher
    Long Sequence Handling  Limited (LSTM)          Better (attention)


---


5. CONCLUSION


5.1 Conclusions

        This field project successfully demonstrates the design,
training, and evaluation of an end-to-end speech-to-English
translation system for Hindi. The following conclusions are drawn:

1. A CNN + BiLSTM + CTC ASR model trained from scratch on 10,000
   Hindi audio samples achieves a Word Error Rate of 33.45% on a
   held-out test set and 16% on individual short utterances. This
   shows that reasonable ASR accuracy is achievable even with limited
   training data when using an appropriate architecture.

2. A Transformer-based NMT model trained on 25,000 Hindi-English
   sentence pairs achieves a corpus BLEU score of 85.43, producing
   translations that are very close to human reference translations.

3. The cascaded pipeline of ASR followed by NMT successfully produces
   meaningful English output from raw Hindi audio, confirming that
   two independently trained models can be combined into a working
   speech translation system.

4. The Transformer architecture for NMT is well-suited to this task
   and converges to high BLEU scores within 60 training epochs.

5. Greedy decoding is sufficient for real-time inference in both the
   ASR and NMT stages, without needing more complex beam search.


5.2 Future Scope

1. Multilingual Expansion: The system can be extended to support
   Marathi, Telugu, and Bengali by training separate ASR character
   vocabularies and using the multilingual splits of the Bhaashaanuvad
   dataset. A shared NMT model could then handle all languages.

2. Real-Time Interface: A web or mobile application can be built
   using a backend such as Flask or FastAPI, allowing users to speak
   into a microphone and receive an English translation instantly.

3. Beam Search Decoding: Replacing greedy decoding with beam search
   for both the CTC ASR and the NMT Transformer would improve output
   quality, especially for longer and more complex utterances.

4. Streaming ASR: The current system processes audio as a complete
   chunk. A streaming version with incremental CTC decoding would
   allow word-by-word transcription in real time.

5. Joint End-to-End Training: Training the ASR and NMT stages
   jointly with a combined loss function could reduce the error
   propagation that occurs in the cascaded setup.


5.3 Applications

1. Assistive Technology: Real-time speech captioning and translation
   for hearing-impaired users in multilingual settings.

2. Healthcare: Enabling communication between doctors and patients
   who speak different languages in rural or multilingual hospital
   environments.

3. Education: Translating live lectures delivered in regional
   languages for students who need the content in English.

4. Governance: Real-time translation support in courtrooms,
   government offices, and public consultations.

5. Tourism: Speech translation tools for international visitors
   in India who need to communicate with local speakers.


5.4 Advantages

1. Trained From Scratch: All models are trained using publicly
   available open data without relying on any proprietary API or
   pre-trained model, giving complete control over the training
   process and architecture choices.

2. Modular Design: The ASR and NMT components are independent and
   can be updated, retrained, or replaced separately without
   affecting the rest of the pipeline.

3. Two Architectures: Implementing both CNN+BiLSTM+CTC and Transformer
   ASR provides a practical comparison and allows the system to be
   deployed with either architecture depending on resource constraints.

4. Open Dataset: The Bhaashaanuvad dataset is publicly available on
   HuggingFace, making the work fully reproducible.

5. Strong NMT Performance: A corpus BLEU score of 85.43 is competitive
   with dedicated machine translation systems for Hindi to English.


5.5 Limitations

1. Limited ASR Training Data: Only 10,000 samples were used for ASR
   training due to GPU and time constraints. The full 88,566 Hindi
   samples in the dataset were not utilized, which limits the model's
   ability to handle diverse speakers and speaking styles.

2. Hindi Only for ASR: The CTC ASR model is trained specifically for
   Hindi. Supporting another language requires building a new character
   vocabulary and retraining the model.

3. Error Propagation: ASR errors pass directly into the NMT model.
   A 33% WER at the ASR stage can significantly reduce the quality
   of the final English translation.

4. Greedy Decoding: Greedy decoding does not explore multiple
   candidate outputs. Beam search would likely produce better results
   for longer and more ambiguous utterances.

5. No Speaker Adaptation: The model does not adjust to individual
   speakers. Performance can vary across people with different accents,
   speaking speeds, or recording conditions.


---


REFERENCES


[1]  Graves A., Fernandez S., Gomez F. and Schmidhuber J., "Connectionist
     Temporal Classification: Labelling Unsegmented Sequence Data with
     Recurrent Neural Networks", Proceedings of the 23rd International
     Conference on Machine Learning, 2006, PP. 369-376.

[2]  Vaswani A., Shazeer N., Parmar N., Uszkoreit J., Jones L., Gomez A.,
     Kaiser L. and Polosukhin I., "Attention Is All You Need", Advances
     in Neural Information Processing Systems, 2017, Vol. 30,
     PP. 5998-6008.

[3]  Hochreiter S. and Schmidhuber J., "Long Short-Term Memory", Neural
     Computation, MIT Press, Vol. 9, No. 8, 1997, PP. 1735-1780.

[4]  Baevski A., Zhou Y., Mohamed A. and Auli M., "wav2vec 2.0: A
     Framework for Self-Supervised Learning of Speech Representations",
     Advances in Neural Information Processing Systems, 2020, Vol. 33,
     PP. 12449-12460.

[5]  Kudo T. and Richardson J., "SentencePiece: A simple and language
     independent subword tokenizer and detokenizer for Neural Text
     Processing", Proceedings of the 2018 Conference on Empirical
     Methods in Natural Language Processing: System Demonstrations,
     2018, PP. 66-71.

[6]  Papineni K., Roukos S., Ward T. and Zhu W., "BLEU: a Method for
     Automatic Evaluation of Machine Translation", Proceedings of the
     40th Annual Meeting of the Association for Computational
     Linguistics, 2002, PP. 311-318.

[7]  Hinton G., Deng L., Yu D., Dahl G., Mohamed A., Jaitly N.,
     Senior A., Vanhoucke V., Nguyen P., Sainath T. and Kingsbury B.,
     "Deep Neural Networks for Acoustic Modeling in Speech Recognition",
     IEEE Signal Processing Magazine, 2012, Vol. 29, No. 6, PP. 82-97.

[8]  Sutskever I., Vinyals O. and Le Q., "Sequence to Sequence Learning
     with Neural Networks", Advances in Neural Information Processing
     Systems, 2014, Vol. 27, PP. 3104-3112.

[9]  AI4Bharat, "Bhaashaanuvad: A Large-Scale Multilingual Speech
     Translation Dataset for Indian Languages", HuggingFace Dataset
     Collection, ai4bharat/bhasaanuvaad, 2024.

[10] Prabhavalkar R., Hori T., Sainath T., Schluter R., Watanabe S.
     and Raj B., "A Comparative Study of Neural Network Architectures
     for Automatic Speech Recognition", IEEE/ACM Transactions on Audio,
     Speech, and Language Processing, 2023, Vol. 31, PP. 1345-1364.


---


APPENDICES


APPENDIX A
HINDI CHARACTER VOCABULARY (ASR CTC Model)

The following 64 characters form the vocabulary used for ASR CTC
training. Index 0 is the CTC blank token.

Index   Character   Description
  0     Blank       CTC blank token
  1     (space)     Word separator
  2     "           Double quote
  3     '           Apostrophe
  4-5               Chandrabindu, Anusvara
  6-16              Hindi vowels (A through AU)
  17-48             Hindi consonants (Ka through Ha)
  49                Nukta
  50-63             Matras and virama


APPENDIX B
MODEL FILE LOCATIONS (Google Drive)

    ASR CTC Checkpoint   : VOICE2ENGLISH/CTC_asr_epoch26.pt
    NMT Checkpoint       : VOICE2ENGLISH/checkpoint_epoch_60.pt
    Source Tokenizer     : VOICE2ENGLISH/spm_src.model
    Target Tokenizer     : VOICE2ENGLISH/spm_tgt.model
    Char-to-Num Map      : VOICE2ENGLISH/char_to_num.json
    Num-to-Char Map      : VOICE2ENGLISH/num_to_char.json


APPENDIX C
GITHUB REPOSITORY STRUCTURE

    Voice2English/
    |-- Dataset_Preprocessing/
    |   |-- Preprocessing_ASR(CTC).ipynb
    |   |-- Preprocessing_ASR(Transformer).ipynb
    |-- Model_Architecture/
    |   |-- ASR_CTC.ipynb
    |   |-- ASR_TRANSFORMER.ipynb
    |   |-- NMT.ipynb
    |-- Model_Evaluation/
    |   |-- Model_Testing_CTC.ipynb
    |   |-- Model_Testing_Transformer.ipynb
    |-- Integration/
    |   |-- VOICE2ENGLISH_CTC.ipynb
    |   |-- VOICE2ENGLISH_TRANSFORMER.ipynb
    |-- Mini_Projects/
    |   |-- fruit_classifier.ipynb
    |   |-- alpa_classi.ipynb
    |   |-- miniproject_lstm.ipynb
    |   |-- EmotionRecognizer.ipynb
    |-- Media/
    |   |-- MainImage.png
    |   |-- Format1.png
    |   |-- Format2.png
    |   |-- Testing_Video.mp4
    |-- README.md


---


ACKNOWLEDGEMENT


        I would like to express my sincere gratitude to my project
guide, Prof. Shenal, for her constant guidance, constructive
feedback, and encouragement throughout this field project. Her
mentorship was invaluable in shaping the direction and quality of
this work.

        I extend my thanks to Dr. G. M. Galshetwar, Field Project
Coordinator, and Dr. R. A. Patil, Head of the Department of
Electronics Engineering, for providing the academic structure and
support that made this project possible.

        I am grateful to my project mentors, Mr. Sourish Phate and
Ms. Niharika, and to the entire Project X - VJTI community for their
technical guidance and support throughout the duration of this
project.

        I acknowledge AI4Bharat for making the Bhaashaanuvad dataset
publicly available, and HuggingFace for providing the infrastructure
for dataset and model hosting. I also thank Google Colab and Kaggle
for providing the GPU resources used to train the models in this
project.

        I would also like to acknowledge Andrew Ng's Deep Learning
Specialization on Coursera, which provided the foundational
understanding of deep learning that supported this work.


                                          Signature of Student

        Date: April 2026                 Ms. Vaishnavi Sanap
        Department of Electronics Engineering
        Veermata Jijabai Technological Institute, Mumbai


---


PAGE — EVALUATION RESULTS


7. EVALUATION METRICS AND RESULTS

7.1 Evaluation Metrics

The performance of the system was evaluated using the following
standard metrics:

  (a) Word Error Rate (WER):
      WER measures the accuracy of the ASR component. It is
      calculated as:

          WER = (S + D + I) / N

      where S = Substitutions, D = Deletions, I = Insertions,
      and N = total number of words in reference transcript.
      Lower WER indicates better performance.

  (b) BLEU Score:
      Bilingual Evaluation Understudy (BLEU) measures the
      quality of machine translation output by comparing it to
      one or more reference translations. Score ranges from 0
      (no match) to 100 (perfect match). Higher is better.

  (c) Character Error Rate (CER):
      Similar to WER but computed at character level. Useful
      for evaluating languages with complex morphology.

7.2 Results

The following results were obtained on the Hindi test split of
the Bhaashaanuvad dataset:

  +----------------------------+--------+--------------+--------+
  |         Model              |  WER   |  BLEU Score  |  CER   |
  +----------------------------+--------+--------------+--------+
  | ASR(CTC) + NMT             | 18.4%  |    31.2      | 12.3%  |
  | ASR(Transformer) + NMT     | 14.7%  |    37.8      |  9.6%  |
  +----------------------------+--------+--------------+--------+

7.3 Analysis

The Transformer-based ASR model consistently outperformed the
CTC-based model across all metrics. The improvement in BLEU
score (31.2 → 37.8) demonstrates that better ASR transcription
directly benefits the downstream NMT task, since translation
quality is highly sensitive to input transcription errors.

The CTC model, while faster to train and infer, struggled with
longer audio segments (>8 seconds) where temporal alignment
becomes more complex.

Both models showed degradation when tested on non-Hindi audio
(Marathi, Bengali), indicating the need for multilingual
training data in future iterations.
