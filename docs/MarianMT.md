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

See [Example Usage](https://github.com/Brycekratzer/NLP-Final-Project-Translatinator/blob/main/docs/MarianMT_Example.ipynb) for using basic model

---

## Evaluation Metrics

There are multiple ways to evaluate a model's performance for Translation. Unlike a classical Machine Learning task where the test set is used to see how well a model performed, translation can be done in multiple ways and still keep a similar meaning that can be interpreted to a good degree given some context. This leads to a more complex way of 
assessing how well some text was translated.

Below are 2 methods that work around this conflict:

### Metrics

- **BLEU Method**

    BLEU (Bilingual Evaluation Understudy) compares machine translations against human references by measuring n-gram overlap. It calculates how many word sequences in the machine output match the reference translations, applies a brevity penalty for short translations, and combines scores from different n-gram sizes (1-4).

    Scores range from 0 to 1, with higher scores indicating better quality. BLEU is valuable for quick, consistent evaluations but doesn't account for synonyms or alternative phrasings that might be equally valid translations. 

    - Basic Usage (Python)

        ```
        predictions = ['Hello! My name is Bryce. How are you?']

        references = ['Hello! My name is Bryce. How are you?', 
                    'Hi! My name's Bryce. How are you?']

        bleu = evalute.load('bleu')

        results = bleu.compute(predictions=predictions, references=references)

        ```

        The code above demonstrates a very basic usage of bleu where 
        - **predictions** is a list of prediction(s)
        - **references** is a list of reference(s) of actual phrases that are similar to the prediction phrases

        The references in this case is how we evaluate the predictions. Thus, we would need high-quality text for our references in order to determine the accuracy and quality of our model.

- **Human Method**

    Human evaluation involves bilingual reviewers assessing translations based on adequacy (correct meaning), fluency (natural-sounding language), grammar, and terminology. Methods include direct scoring, ranking different translations, and detailed error analysis.
    While more expensive and time-consuming than automated metrics, human evaluation captures nuances that BLEU cannot, including cultural appropriateness and contextual accuracy. Most robust evaluation approaches combine both BLEU for efficiency and human review for deeper quality insights.

---

## Resources and References

### Official Documentation
- [MarianMT Hugging Face Documentation](https://huggingface.co/docs/transformers/en/model_doc/marian)

### Example Implementations

### Additional Learning Materials
- [Building a Simple Language Translation Tool Using a Pre-Trained Translation Model - GeeksforGeeks](https://www.geeksforgeeks.org/building-a-simple-language-translation-tool-using-a-pre-trained-translation-model/#1-marianmt)

---

*This documentation was prepared by Bryce Kratzer for The Translatinator project.*