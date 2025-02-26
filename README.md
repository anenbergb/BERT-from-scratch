# BERT-from-scratch
Implementation of BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding



Tokenizer
* https://github.com/google/sentencepiece


# References
* BERT paper https://arxiv.org/abs/1810.04805
* Blog post https://jalammar.github.io/illustrated-bert/

* https://www.philschmid.de/pre-training-bert-habana
* Huggingface BERT model https://huggingface.co/google-bert/bert-base-uncased
* Huggingface https://huggingface.co/docs/transformers/model_doc/bert




# Dataset

BooksCorpus  
* A dataset containing over 11,000 unpublished books scraped from the web, covering a variety of genres. It includes approximately 800 million words.
* HuggingFace dataset https://huggingface.co/datasets/bookcorpus/bookcorpus.
English Wikipedia
* A cleaned-up version of English Wikipedia articles, consisting of about 2.5 billion words. Only the text content was used (lists, tables, and headers were excluded).
* Huggingface dataset https://huggingface.co/datasets/wikimedia/wikipedia. 

The following are the number of words I found to exist in either dataset
```
BooksCorpus # words: 840.01M
Wikipedia (English) # words: 3.17B
Total # words: 4.01B
```
It appears that the total corpusu size is 4.01B words rather than 3.3B words, as reported in the [BERT paper](https://arxiv.org/abs/1810.04805)

# Tokenization

BERT uses WordPiece, a subword tokenization method, as its tokenizer. This approach strikes a balance between word-level and character-level tokenization, allowing BERT to efficiently handle a wide vocabulary while managing rare or out-of-vocabulary words.
Details of WordPiece in BERT:
* How It Works: WordPiece breaks down text into smaller units (subwords or word pieces) based on a pre-trained vocabulary. It starts with individual characters and iteratively merges them into larger tokens (e.g., "playing" might be split into "play" and "##ing"), guided by a likelihood-based algorithm that maximizes the probability of the training corpus.
* Vocabulary Size: BERT’s WordPiece tokenizer has a vocabulary of 30,000 tokens, which includes whole words, subwords, and special tokens like [CLS] (for classification) and [SEP] (to separate sentences).
* Special Tokens: 
```
[CLS]: Added at the beginning of every input sequence for classification tasks.
[SEP]: Used to separate sentences in tasks like Next Sentence Prediction or to mark the end of a single sequence.
[PAD]: Used for padding shorter sequences to match the maximum length.
[MASK]: Used during pre-training for the Masked Language Modeling task.
##: A prefix for subword pieces that attach to a previous token (e.g., "playing" → "play" + "##ing").
```
Training: The WordPiece vocabulary was trained on the same datasets as BERT (BooksCorpus and English Wikipedia), ensuring it reflects the statistical properties of the pre-training corpus.

Why WordPiece?
* Efficiency: It reduces the vocabulary size compared to full-word tokenization, making it computationally manageable.
* Flexibility: It can handle unseen words by breaking them into known subwords (e.g., "unhappiness" → "un" + "##happi" + "##ness").
* Bidirectionality: It pairs well with BERT’s bidirectional architecture, as it preserves meaningful chunks of text for contextual understanding.


Tokenizer web app
* https://tiktokenizer.vercel.app/


Byte Pair Encoding
- https://huggingface.co/learn/nlp-course/en/chapter6/5
- Get the unique set of words in the corpus
- Build the vocabulary by taking all the symbols to write those words. e.g. all the ASCII and Unicode characters
- Add new tokens to the vocabulary by merging 2 items in the vocabulary. The pair of tokens to merge is determined by identifying the pair of tokens that most commonly occur next to eachother.  

WordPiece tokenizer
* https://huggingface.co/learn/nlp-course/en/chapter6/6?fw=pt
* https://huggingface.co/learn/nlp-course/en/chapter6/8?fw=pt


Cased vs. Uncased
* Uncased means that the text has been lowercased before WordPiece tokenization, e.g., John Smith becomes john smith. The Uncased model also strips out any accent markers.
* Cased means that the true case and accent markers are preserved. Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging).

Tokenization steps
1. Normalization (any cleanup of the text that is deemed necessary, such as removing spaces or accents, Unicode normalization, etc.)
2. Pre-tokenization (splitting the input into words)
3. Running the input through the model (using the pre-tokenized words to produce a sequence of tokens)
4. Post-processing (adding the special tokens of the tokenizer, generating the attention mask and token type IDs)

## Huggingface Tokenizer
[BertTokenizer](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertTokenizer)
* BERT tokenizer baed on WordPiece
[BertTokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertTokenizerFast)
* "fast" BERT tokenizer based on WordPiece


# Masking
* Whole word masking is preferred to masking individual WordPiece tokens https://github.com/google-research/bert/blob/master/README.md

# Training hyperparameters


## Pre-Training: Original BERT settings
```
Model Variants:
    BERT-Base: 12 layers, 768 hidden size, 12 attention heads, 110M parameters.
    BERT-Large: 24 layers, 1024 hidden size, 16 attention heads, 340M parameters.

Batch Size: 256 sequences (split across multiple GPUs).
Learning Rate: 1e-4 (with Adam optimizer).
Optimizer: Adam with β1 = 0.9, β2 = 0.999, ε = 1e-6.
Learning Rate Schedule: Linear warmup over the first 10,000 steps, followed by linear decay.
Training Steps: 1,000,000 steps (about 40 epochs over 3.3 billion words).
Max Sequence Length: 128 tokens for the first 90% of steps, then 512 tokens for the last 10%.
Dropout: 0.1 on all layers.
Weight Decay: 0.01.
Loss: Masked Language Model (MLM) + Next Sentence Prediction (NSP), with 15% of tokens masked for MLM.
```
* coudl try reducing max steps to 100k


## Fine-Tuning
```
Batch Size: 
    16 or 32 (BERT-Base);
    8 or 16 (BERT-Large).
    Adjust based on GPU memory (e.g., 16GB V100 can handle batch size 32 for BERT-Base with max sequence length 128).

Learning Rate: 2e-5 (common starting point), with options of 3e-5 or 5e-5 depending on task.
    Smaller datasets may benefit from lower rates (e.g., 1e-5); larger datasets can handle slightly higher rates.

Optimizer: AdamW (Adam with weight decay fix)
    β1 = 0.9, β2 = 0.999, ε = 1e-8 (slightly adjusted from pre-training for stability).
Weight Decay: 0.01 (applied to all parameters except bias and LayerNorm).
Learning Rate Schedule: Linear warmup over the first 10% of steps, followed by linear decay to 0.
Warmup Steps: 0–500 (typically 10% of total steps or a fixed 100–500 for smaller datasets).
Epochs: 2–4 (often 3 is sufficient; overtraining beyond 4 can lead to overfitting on small datasets).
Max Sequence Length:
    128 (for most tasks like classification)
    512 (for tasks like question answering with long contexts).
    Truncate or pad inputs accordingly.
Dropout: 0.1 (unchanged from pre-training).
```

## Fine-Tuning Task-Specific Guidance
Text Classification (e.g., Sentiment Analysis, GLUE Tasks)
```
Batch Size: 16 or 32.
Learning Rate: 2e-5 or 3e-5.
Epochs: 3.
Example: For GLUE benchmarks, the original paper reports fine-tuning BERT-Base with these settings achieves strong results (e.g., 84.6% on MNLI).
```
Question Answering (e.g., SQuAD)
```
Batch Size: 12–16 (due to longer sequences).
Learning Rate: 3e-5 or 5e-5.
Max Sequence Length: 384 or 512.
Epochs: 2–3.
Example: BERT-Large fine-tuned on SQuAD 2.0 with LR=5e-5, batch size=12, 3 epochs achieves 91.0 F1.
```
Named Entity Recognition (NER)
```
Batch Size: 16.
Learning Rate: 2e-5.
Epochs: 3–5 (NER datasets are often smaller, so monitor validation loss).
Max Sequence Length: 128.
Small Datasets (<10k Samples):
Learning Rate: Lower to 1e-5 or 2e-5 to avoid overfitting.
Epochs: 2–3 (use early stopping).
Consider freezing lower layers and only fine-tuning the top layers or classifier head.
```

* gradient accumulation shoudl be used to achieve larger effective batch size
* early stopping may be performed to halt training if performance plateaus