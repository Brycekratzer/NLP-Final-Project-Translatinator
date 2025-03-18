# MarianMT Translation Model Research
## The Translatinator Project

## Table of Contents
1. [Basic Understanding of MarianMT](#basic-understanding-of-marianmt)
2. [Neural Machine Translation Architecture](#neural-machine-translation-architecture)
3. [Integration with Speech Recognition](#integration-with-speech-recognition)
4. [Implementation Guidelines](#implementation-guidelines)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Resources and References](#resources-and-references)

---

## Basic Understanding of MarianMT

### What is MarianMT?
- MarianMT is a collections of pre-trained translation models that support wide range of 
language pairs. 

- Developed by Microsoft Translator team. 

- Based on Marian neural machine translation architecture
    - Designed for high-quality translation across numerous languages

### Pre-trained Models
- All Models use the following format
    - `Helsinki-NLP/opus-mt-{source language}-{target language}`
- **Model For This Case** 
    - `Helsinki-NLP/opus-mt-en-es` 
        
    - [English to Spanish Model Documentation](https://huggingface.co/Helsinki-NLP/opus-mt-en-es)

### Core Components
- The Tokenizor uses a tool called **SentencePiece**
    - Breaks up portions of words from a language into a sub-words
        - Example:

        If we had 'untranslatable' in our sentence to translate SentencePiece would split it up into

        '__un'

        'translat'

        'able'
    - Sentence Piece is a trained model that learns how to break down text 
    based on statisitcal patterns

    - Can rebuild words from these pieces

- TODO

---

## Neural Machine Translation Architecture

### TODO

---

## Integration with Speech Recognition

### TODO

---

## Implementation Guidelines

### Setting Up MarianMT with Hugging Face
- Required libraries and dependencies
    - **transformers** Hugging Face Library
        - MarianMTModel
        - MarianTokenizer
    - **SentencePiece** dependency for MarianTokenizer

### Basic Usage

See [Example Usage](docs/MarianMT_Example.ipynb) for using basic model

---

## Evaluation Metrics

### TODO

---

## Resources and References

### Official Documentation
- [MarianMT Hugging Face Documentation](https://huggingface.co/docs/transformers/en/model_doc/marian)

### Example Implementations

### Additional Learning Materials
- [Building a Simple Language Translation Tool Using a Pre-Trained Translation Model - GeeksforGeeks](https://www.geeksforgeeks.org/building-a-simple-language-translation-tool-using-a-pre-trained-translation-model/#1-marianmt)

---

*This documentation was prepared by Bryce Kratzer for The Translatinator project.*