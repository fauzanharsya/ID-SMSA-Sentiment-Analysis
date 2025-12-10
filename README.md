# Evaluating Performance and Computational Efficiency: Benchmarking Hybrid IndoBERT Models for Indonesian Financial Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fauzanharsya/ID-SMSA-Sentiment-Analysis/blob/main/ID_SMSA_Sentiment_Benchmark.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20HuggingFace-orange)](https://tensorflow.org/)

This repository contains the official implementation code for our research on financial sentiment analysis in the Indonesian stock market. We provide a comprehensive benchmark of conventional Deep Learning models vs. Transformer-based models and introduce a resource-efficient **Hybrid-IndoBERT** architecture.

## 📌 Abstract

Financial sentiment analysis on social media is critical for market prediction, yet it faces challenges in low-resource languages like Indonesian due to linguistic complexity and high computational costs. This study establishes a comprehensive benchmark on the **ID-SMSA dataset** by evaluating three conventional models (LSTM, GRU, CNN) and two Transformer-based strategies (IndoBERT and BERT). Additionally, we propose a resource-efficient **Hybrid-IndoBERT** architecture to address the performance-efficiency trade-off.

**Key Findings:**
*   **Performance:** Fine-tuned IndoBERT achieved state-of-the-art performance (**81.46% Accuracy**), outperforming English-translated BERT.
*   **Efficiency:** The proposed **Hybrid model** achieved competitive accuracy (**79.94%**) while reducing GPU VRAM consumption by approximately **75%** compared to full fine-tuning.

## 🛠️ Methodology & Experimental Setup

We implemented a rigorous pipeline to ensure fair comparison:

1.  **Data Preprocessing:**
    *   Text Cleaning (Regex).
    *   **Domain-Specific Normalization:** Implemented a **Whitelist Mechanism** to protect stock tickers (e.g., `BBCA`, `GOTO`) and financial slang (e.g., `cuan`, `boncos`) from incorrect normalization.
    *   **Handling Imbalance:** Applied **Class Weighting** to the loss function.
    
2.  **Models Evaluated:**
    *   **Conventional:** Bidirectional LSTM, Bidirectional GRU, CNN (Conv1D).
    *   **Transformer (End-to-End):** IndoBERT-base-p1 (Native ID) vs. BERT-base-uncased (Translated EN).
    *   **Hybrid (Proposed):** Frozen IndoBERT Backbone + BiGRU + Attention Mechanism.

## 📊 Benchmark Results

Below is the comprehensive summary of our experimental results, comparing conventional Deep Learning models and Transformer-based models across native and translated datasets. Sorted by **Macro F1-Score**.

| Model Strategy | Architecture | Input Language | Test Acc | Macro F1 | Training Time (s) | VRAM Usage (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IndoBERT (Fine-Tuning)** | Transformer | Native (ID) | **0.815** | **0.797** | 217 | 4212 |
| **BERT (English Fine-Tuning)** | Transformer | Translated (EN) | 0.795 | 0.772 | 207 | 4212 |
| **Hybrid IndoBERT (Frozen)** | **Hybrid** | **Native (ID)** | 0.799 | 0.767 | 235 | **1046** 📉 |
| CNN (Indonesian) | Conventional | Native (ID) | 0.752 | 0.723 | 10 | ~250 |
| GRU (Indonesian) | Conventional | Native (ID) | 0.729 | 0.703 | 13 | ~250 |
| LSTM (Indonesian) | Conventional | Native (ID) | 0.725 | 0.702 | 17 | ~250 |
| CNN (English) | Conventional | Translated (EN) | 0.713 | 0.674 | 8 | ~250 |
| LSTM (English) | Conventional | Translated (EN) | 0.699 | 0.667 | 13 | ~250 |
| GRU (English) | Conventional | Translated (EN) | 0.696 | 0.664 | 14 | ~250 |

> **Notes:**
> *   **VRAM Usage:** For conventional models, values are estimated based on peak active memory (~250MB) as TensorFlow's greedy allocation often reports maximum available memory in logs.
> *   **Performance:** Transformer models significantly outperform conventional baselines.
> *   **Efficiency:** The **Hybrid IndoBERT** model reduces VRAM usage by **~75%** compared to full fine-tuning, offering a viable trade-off for resource-constrained environments.

## 🚀 How to Run

The easiest way to reproduce our experiments is by running the notebook in Google Colab.

1.  Click the **"Open in Colab"** badge at the top of this README.
2.  Ensure you are connected to a **GPU Runtime** (Runtime > Change runtime type > T4 GPU).
3.  Run all cells sequentially. The notebook handles data downloading, preprocessing, training, and evaluation automatically.

## 📦 Requirements

If you prefer running it locally, ensure you have the following libraries installed:

```bash
pip install tensorflow transformers scikit-learn pandas numpy matplotlib seaborn deep-translator
```

## 👥 Authors

*   **Fauzan Aditya Harsya**
*   **Daniel Grely Wirawan**
*   **Azhar Alzaki Rosanto**
*   **Zahra Nabila Izdihar**
*   **Said Achmad**


*Computer Science Department, Bina Nusantara University*

