<h1 align="center"> Voice 2 English 🗣️</h1>

<p align="center">
  <img src="Media/MainImage.png" alt="voice2english" width="350" />
</p>

<div align="center">
  <strong>The goal is to convert any spoken language into transcript of that original language and then convert that transcript into english language.</strong>
</div>

<br />

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Results](#-results)
- [Tech Stack](#️-tech-stack)
- [File Structure](#-file-structure)
- [Dataset](#-dataset-miracl-vc1)
- [Model Architecture](#-model-architecture)
- [Installation and Setup](#-installation-and-setup)
- [Future Scope](#-future-scope)
- [Acknowledgements](#-acknowledgement)
- [Contributors](#-contributors)

## 📘 About the Project

This project focuses on developing an advanced voice-to-English translation system that converts spoken input from any language into English text. The pipeline begins with Automatic Speech Recognition (ASR) models, which transcribe the spoken language into its original text form. Next, a Neural Machine Translation (NMT) model translates this transcript into fluent and contextually accurate English. The dataset is divided into training and testing sets to ensure fair and consistent evaluation. The system leverages modern deep learning architectures optimized through extensive hyperparameter tuning to maximize transcription and translation accuracy. Finally, the project includes a real-time speech translation feature, enabling seamless live audio input and instant English output.

## 📊 Results

### Live Deteection


### Online Testing


### Confusion Matrix


### Accuracy


## ⚙️ Tech Stack


## 📁 File Structure


## 💾 Dataset: Bhaashaanuvad

The **Bhaashaanuvad** dataset is designed to support research in multilingual speech-to-text and translation systems. It focuses on converting spoken audio from various Indian and global languages into accurate transcriptions and their corresponding English translations. Below is a breakdown of its structure and contents:

🎙️ Audio Samples: Contains recordings of spoken sentences in multiple languages (such as Hindi, Marathi, Bengali, Tamil, and others).

📝 Transcriptions: Each audio file is paired with a text transcript of the original language, facilitating ASR (Automatic Speech Recognition) training.

🌐 Translations: Provides parallel English translations of the transcripts, enabling effective NMT (Neural Machine Translation) training.

🧠 Purpose: Built to train and evaluate end-to-end speech translation pipelines, particularly systems that integrate ASR and NMT models for real-time multilingual voice translation.

⚙️ Format: Data is organized in JSON and CSV formats, with fields for audio_path, source_text, and translated_text.

[Download the Bhaashaanuvad dataset on HuggingFace](https://huggingface.co/collections/ai4bharat/bhasaanuvaad-672b3790b6470eab68b1cb87)


