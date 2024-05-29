# LLM Prompt Recovery

This repository contains the code and resources for the project titled "LLM Prompt Recovery," an ECE 5424 final project. The project aims to develop methodologies for recovering the original prompts used in text rewriting tasks performed by large language models (LLMs).

## Table of Contents

- [Introduction](#introduction)
- [Dataset Compilation](#dataset-compilation)
  - [Original Text Selection](#original-text-selection)
  - [Prompt Generation](#prompt-generation)
  - [Rewritten Text Generation](#rewritten-text-generation)
  - [Dataset Composition and Splitting](#dataset-composition-and-splitting)
- [Methodology](#methodology)
  - [Preprocessing](#preprocessing)
  - [Models](#models)
    - [Mixtral and Mistral 7B](#mixtral-and-mistral-7b)
    - [LLaMA2 7B](#llama2-7b)
    - [Gemma 2B and Gemma 7B](#gemma-2b-and-gemma-7b)
    - [T5 and BART](#t5-and-bart)
- [Experimental Setup and Evaluation](#experimental-setup-and-evaluation)
  - [Zero-Shot Learning](#zero-shot-learning)
  - [Few-Shot Learning](#few-shot-learning)
  - [Fine-Tuning Transformer Models](#fine-tuning-transformer-models)
  - [Evaluation Metrics](#evaluation-metrics)
- [Results and Discussion](#results-and-discussion)
  - [Zero-Shot Learning Performance](#zero-shot-learning-performance)
  - [Few-Shot Learning Performance](#few-shot-learning-performance)
  - [Fine-Tuning Transformer Models Performance](#fine-tuning-transformer-models-performance)
  - [Comparative Analysis and Discussion](#comparative-analysis-and-discussion)
- [Conclusion](#conclusion)
- [Appendix](#appendix)
  - [Detailed Prompts for Model Prediction](#detailed-prompts-for-model-prediction)
  - [Qualitative Results](#qualitative-results)

## Introduction

The project explores the relationship between input prompts and the generated outputs in LLMs, focusing on text rewriting tasks. The primary objectives are:
1. Develop techniques to recover original prompts.
2. Gain insights into LLMs' interpretation of prompts.
3. Compare different prompt recovery approaches.

## Dataset Compilation

### Original Text Selection

The original texts were selected from the summary attribute of the Wikipedia movie plots dataset, offering a diverse range of genres and styles.

### Prompt Generation

Prompts were generated using the Claude 3 Opus model. Ten categories of prompts were defined, including content modification, cultural adaptations, emotion and sentiment changes, and more.

### Rewritten Text Generation

The rewritten texts were generated using the Gemma 7B-it model. Each original text was paired with multiple prompts to introduce variability.

### Dataset Composition and Splitting

The dataset comprises 6,000 samples, split into a training set (4,800 samples) and a test set (1,200 samples).

## Methodology

### Preprocessing

Key preprocessing steps include text cleaning, tokenization, sequence length handling, prompt formatting, and train-test splitting.

### Models

#### Mixtral and Mistral 7B

These models use a combination of autoregressive and bidirectional language modeling.

#### LLaMA2 7B

This model has been trained on a massive corpus of text data and shows strong language understanding capabilities.

#### Gemma 2B and Gemma 7B

These models have been trained on extensive text data and are evaluated for their performance in zero-shot, few-shot, and fine-tuning settings.

#### T5 and BART

These transformer-based models are fine-tuned for the prompt recovery task.

## Experimental Setup and Evaluation

### Zero-Shot Learning

In this setting, models are evaluated without additional training. Beam search decoding is used for prompt generation.

### Few-Shot Learning

Models are provided with a limited number of training examples to adapt and generate prompts.

### Fine-Tuning Transformer Models

Models are fine-tuned on the training dataset to improve prompt recovery performance.

### Evaluation Metrics

Performance is assessed using Cosine Similarity and ROUGE Score.

## Results and Discussion

### Zero-Shot Learning Performance

LLaMA2 7B showed the highest performance in the zero-shot setting.

### Few-Shot Learning Performance

All models improved with few-shot examples, with LLaMA2 7B leading the performance.

### Fine-Tuning Transformer Models Performance

Fine-tuning yielded the highest performance, with Gemma 7B achieving the best results.

### Comparative Analysis and Discussion

Performance trends show improvements with task-specific training. Fine-tuning provides the best results but requires more resources.

## Conclusion

The project successfully demonstrates the feasibility of prompt recovery using LLMs. Fine-tuning transformer models yields the best performance, contributing to more interpretable and controllable language models.

## Appendix

### Detailed Prompts for Model Prediction

Examples of zero-shot and few-shot learning prompts used for model predictions.

### Qualitative Results

Examples of qualitative results for zero-shot, few-shot, and fine-tuning settings.

## References

Refer to the report for a detailed list of references.

---

For more details, please refer to the project report included in this repository.
