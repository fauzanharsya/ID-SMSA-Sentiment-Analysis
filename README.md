# Evaluating Performance and Computational Efficiency: Benchmarking Hybrid IndoBERT Models for Indonesian Financial Sentiment Analysis

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fauzanharsya/ID-SMSA-Sentiment-Analysis/blob/main/ID_SMSA_Sentiment_Benchmark.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20HuggingFace-orange)](https://tensorflow.org/)

This repository contains the official implementation code for our research on financial sentiment analysis in the Indonesian stock market. We provide a comprehensive benchmark of conventional Deep Learning models vs. Transformer-based models and introduce a resource-efficient **Hybrid-IndoBERT** architecture.

## 📌 Abstract

Financial sentiment analysis on social media is critical for market prediction, yet it faces challenges in low-resource languages like Indonesian due to linguistic complexity and high computational costs. This study establishes a comprehensive benchmark on the **ID-SMSA dataset** by evaluating three conventional models (LSTM, GRU, CNN) and three Transformer-based strategies (IndoBERT, BERT, and XLM-RoBERTa). Additionally, we propose a resource-efficient **Hybrid-IndoBERT** architecture to address the performance-efficiency trade-off.

**Key Findings:**
*   **Performance:** Fine-tuned IndoBERT achieved the best performance (**82.83% Accuracy**, Macro F1 = 0.808), outperforming all other models including English BERT and multilingual XLM-RoBERTa.
*   **Efficiency:** The proposed **Hybrid model** achieved competitive accuracy (**81.46%**, Macro F1 = 0.791) while reducing GPU VRAM consumption by approximately **86%** compared to full fine-tuning.

## 🛠️ Methodology & Experimental Setup

We implemented a rigorous pipeline to ensure fair comparison:

1.  **Data Preprocessing:**
    *   Text Cleaning (Regex) with consistent pipeline for both Indonesian and English text.
    *   **Domain-Specific Normalization:** Implemented a **Whitelist Mechanism** to protect stock tickers (e.g., `BBCA`, `GOTO`) and financial slang (e.g., `cuan`, `boncos`) from incorrect normalization.
    *   **Handling Imbalance:** Applied **Class Weighting** to the loss function.
    
2.  **Models Evaluated:**
    *   **Conventional:** Bidirectional LSTM, Bidirectional GRU, CNN (Conv1D).
    *   **Transformer (End-to-End):** IndoBERT-base-p1 (Native ID), BERT-base-uncased (Translated EN), XLM-RoBERTa-base (Native ID, Multilingual).
    *   **Hybrid (Proposed):** Frozen IndoBERT Backbone + BiGRU + Attention Mechanism.

3.  **VRAM Measurement:**
    *   Conventional models are evaluated in **isolated Python subprocesses** to prevent TensorFlow memory caching artifacts.
    *   All models use `tf.config.experimental.get_memory_info('GPU:0')['peak']` for consistent VRAM measurement.

## 📊 Benchmark Results

Below is the comprehensive summary of our experimental results. Sorted by **Macro F1-Score**.

| Model Strategy | Architecture | Input Language | Test Acc | Macro F1 | Training Time (s) | VRAM Usage (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **IndoBERT (Fine-Tuning)** | Transformer (E2E) | Native (ID) | **0.828** | **0.808** | 228 | 3909 |
| **Hybrid IndoBERT (Frozen)** | **Hybrid** | **Native (ID)** | 0.815 | 0.791 | 243 | **561** 📉 |
| XLM-RoBERTa | Transformer (E2E) | Native (ID) | 0.802 | 0.782 | 235 | 7847 |
| BERT (English Fine-Tuning) | Transformer (E2E) | Translated (EN) | 0.801 | 0.780 | 222 | 3763 |
| CNN (Indonesian) | Conventional | Native (ID) | 0.764 | 0.737 | 7 | 217 |
| CNN (English) | Conventional | Translated (EN) | 0.749 | 0.722 | 9 | 217 |
| LSTM (Indonesian) | Conventional | Native (ID) | 0.745 | 0.715 | 13 | 50 |
| GRU (Indonesian) | Conventional | Native (ID) | 0.729 | 0.695 | 13 | 49 |
| LSTM (English) | Conventional | Translated (EN) | 0.729 | 0.689 | 13 | 39 |
| GRU (English) | Conventional | Translated (EN) | 0.704 | 0.662 | 12 | 50 |

> **Notes:**
> *   **VRAM Usage:** All values are actual peak measurements using `tf.config.experimental.get_memory_info('GPU:0')['peak']`. Conventional models use subprocess isolation for accurate measurement.
> *   **Performance:** Transformer models significantly outperform conventional baselines. Native Indonesian models consistently outperform English-translated counterparts.
> *   **Efficiency:** The **Hybrid IndoBERT** model reduces VRAM usage by **~86%** compared to full fine-tuning, offering a viable trade-off for resource-constrained environments.

## 🚀 How to Run

The easiest way to reproduce our experiments is by running the notebook in Google Colab.

1.  Click the **"Open in Colab"** badge at the top of this README.
2.  Ensure you are connected to a **GPU Runtime** (Runtime > Change runtime type > T4 GPU).
3.  Run all cells sequentially. The notebook handles data downloading, preprocessing, training, and evaluation automatically.

## 📦 Requirements

If you prefer running it locally, ensure you have the following libraries installed:

```bash
pip install tensorflow transformers scikit-learn pandas numpy matplotlib seaborn
```

## 👥 Authors

*   **Fauzan Aditya Harsya**
*   **Daniel Grely Wirawan**
*   **Azhar Alzaki Rosanto**
*   **Zahra Nabila Izdihar**
*   **Said Achmad**

*Computer Science Department, Bina Nusantara University*
