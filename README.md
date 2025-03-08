# BERT-from-scratch
Implements "Masked Language Model" pre-training for [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

## 1M Iteration Training Results
I successfully pre-trained BERT-base model for 1M iterations using the randomized whole word masking procedure.

The BERT-base trained on 128 token length sequences achieved the following loss and perplexity.
|            | Train | Test  |
|------------|-------|-------|
| Loss       | 2.065 | 2.391 |
| Perplexity | 7.883 | 10.93 |

<img src="https://github.com/user-attachments/assets/cbd2b5b8-6c38-49a0-8019-e847362721d1" width="500"/>

10k step linear warm-up and linear decay for the subsequent 990k iterations was used.

<img src="https://github.com/user-attachments/assets/743d3dab-24f0-4b49-86ed-e563636f2c7e" width="300"/>

### Example Masked Token Predictions
5% of the dataset is reserved for the validation set. Validation is performed every 100k iterations during training.
Sequences and masked predictions are randomly sampled from the validation set and logged to tensorboard such that the
model performance can be qualitatively accessed.

<img src="https://github.com/user-attachments/assets/de5c724e-8e36-47ad-afaf-ba748076dbf0" width="700"/>


Some example results are displayed below
```
# Easy example, top 1 predictions are correct
text:           neurensin - 2 is a protein that in humans is encoded by the nrsn2 gene. references further reading
text_with_mask: neurensin - 2 [MASK] a protein [MASK] [MASK] humans is encoded [MASK] [MASK] nrsn2 gene. references [MASK] reading
pred_top_1:     neurensin - 2 is[98.9%] a protein that[97.6%] in[98.6%] humans is encoded by[100.0%] the[99.8%] nrsn2 gene. references further[100.0%] reading
pred_top_2:     neurensin - 2,[1.0%] a protein which[1.9%] to[1.0%] humans is encoded as[0.0%] an[0.1%] nrsn2 gene. references related[0.0%] reading
pred_top_3:     neurensin - 2 -[0.1%] a protein found[0.2%] on[0.1%] humans is encoded in[0.0%] a[0.0%] nrsn2 gene. references additional[0.0%] reading

# Correct verb prediction in the top 3 predictions
text:           they let me wait until you were awake.
text_with_mask: they let me [MASK] until you were awake.
pred_top_1:     they let me stay[17.3%] until you were awake.
pred_top_2:     they let me go[14.5%] until you were awake.
pred_top_3:     they let me wait[13.2%] until you were awake.

# More challenging example with adjacent masks. However, the top 3 predictions are still reasonable
text:           she coughs a few more times before picking up her napkin and wiping her face.
text_with_mask: she [MASK] [MASK] a [MASK] more times before picking up her napkin [MASK] wiping her face.
pred_top_1:     she blink[17.1%]s[21.5%] a few[78.6%] more times before picking up her napkin and[99.3%] wiping her face.
pred_top_2:     she sniff[14.8%]ed[7.4%] a couple[20.7%] more times before picking up her napkin,[0.6%] wiping her face.
pred_top_3:     she puff[3.2%] up[5.3%] a dozen[0.2%] more times before picking up her napkin before[0.0%] wiping her face.

# Very challenging example with multiple consecutive masks.
text:           i looked into her tricolored eyes ; a ring of blue, silver, and an inner ring of lights as if light could be a color.
text_with_mask: i looked into [MASK] [MASK] [MASK] [MASK] eyes ; a ring of blue, silver, and an inner [MASK] of lights as if light [MASK] be a color [MASK]
pred_top_1:     i looked into the[19.4%] un[1.4%] '[8.9%] s[13.4%] eyes ; a ring of blue, silver, and an inner ring[23.9%] of lights as if light might[38.3%] be a color.[99.8%]
pred_top_2:     i looked into his[18.4%] '[1.3%] with[5.4%]s[6.5%] eyes ; a ring of blue, silver, and an inner circle[21.2%] of lights as if light could[31.9%] be a color![0.1%]
pred_top_3:     i looked into her[11.0%],[1.1%],[3.2%] blue[4.2%] eyes ; a ring of blue, silver, and an inner set[8.9%] of lights as if light would[9.9%] be a color?[0.1%]

# Accurate predictions across a longer sequence
text:           the vietnamese people in france (, ) consists of people of full or partially vietnamese ancestry who were born in or immigrated to france. their population was about 400, 000 as of 2017, making them one of the largest asian communities in the country. unlike other overseas vietnamese communities in the west, the vietnamese population in france had already been well - established before the fall of saigon and the resulting diaspora. they make up over half of the vietnamese population in europe. history before 1954 france was the first western country in which vietnamese migrants settled due to the colonization of vietnam by france. the french assistance to nguyen anh in 1777 was one of
text_with_mask: the vietnamese people in france (, [MASK] [MASK] of people of full or partially vietnamese ancestry who were born in or [MASK] to france. their population was about 400, 000 as of 2017, making [MASK] one of the [MASK] asian [MASK] in the country. unlike other overseas vietnamese [MASK] in [MASK] west, the [MASK] [MASK] in france had already been well [MASK] established before the fall of saigon and the resulting diaspora. they [MASK] up over half of the vietnamese population in europe. history before 1954 france was [MASK] first western country in which vietnamese migrants settled [MASK] to the colonization of [MASK] by france. the [MASK] assistance to nguyen [MASK] [MASK] in 1777 was one [MASK]
pred_top_1:     the vietnamese people in france (, )[91.4%] consist[51.2%] of people of full or partially vietnamese ancestry who were born in or immigrated[41.8%] to france. their population was about 400, 000 as of 2017, making them[78.3%] one of the largest[83.1%] asian populations[18.3%] in the country. unlike other overseas vietnamese populations[27.4%] in the[99.8%] west, the vietnamese[97.6%] people[47.5%] in france had already been well -[99.8%] established before the fall of saigon and the resulting diaspora. they make[52.8%] up over half of the vietnamese population in europe. history before 1954 france was the[100.0%] first western country in which vietnamese migrants settled prior[71.7%] to the colonization of indochina[50.1%] by france. the french[44.9%] assistance to nguyen đ[17.4%]h[15.0%] in 1777 was one of[95.5%]
pred_top_2:     the vietnamese people in france (, are[4.7%] consists[33.5%] of people of full or partially vietnamese ancestry who were born in or migrated[13.5%] to france. their population was about 400, 000 as of 2017, making it[16.2%] one of the wealthiest[4.5%] asian communities[17.9%] in the country. unlike other overseas vietnamese communities[20.8%] in french[0.0%] west, the vietnam[0.4%] population[11.3%] in france had already been well well[0.0%] established before the fall of saigon and the resulting diaspora. they made[46.2%] up over half of the vietnamese population in europe. history before 1954 france was a[0.0%] first western country in which vietnamese migrants settled due[24.0%] to the colonization of vietnam[23.8%] by france. the vietnamese[12.7%] assistance to nguyen tr[7.8%]yen[13.3%] in 1777 was one that[0.6%]
pred_top_3:     the vietnamese people in france (, is[2.9%] consisted[4.7%] of people of full or partially vietnamese ancestry who were born in or emigrated[11.8%] to france. their population was about 400, 000 as of 2017, making france[4.0%] one of the few[3.8%] asian countries[16.8%] in the country. unlike other overseas vietnamese people[11.2%] in its[0.0%] west, the chinese[0.4%]s[7.5%] in france had already been well so[0.0%] established before the fall of saigon and the resulting diaspora. they makes[0.8%] up over half of the vietnamese population in europe. history before 1954 france was its[0.0%] first western country in which vietnamese migrants settled thanks[1.8%] to the colonization of china[5.4%] by france. the military[7.0%] assistance to nguyen ph[7.0%]nh[6.5%] in 1777 was one in[0.2%]

text:           faster ( hyperpnea ). the exact degree of hyperpnea is determined by the blood gas homeostat, which regulates the partial pressures of oxygen and carbon dioxide in the arterial blood. this homeostat prioritizes the regulation of the arterial partial pressure of carbon dioxide over that of oxygen at sea level. that is to say, at sea level the arterial partial pressure of co2 is maintained at very close to 5. 3 kpa ( or 40 mmhg ) under a wide range of circumstances, at the expense of the arterial partial pressure of o2, which is allowed to vary within a very wide range of
text_with_mask: faster ( hyperpnea ). the exact degree of hyperpnea [MASK] determined by the blood gas homeostat, which regulates the partial [MASK] of oxygen and carbon dioxide in [MASK] arterial blood. this homeostat prioritizes the [MASK] of [MASK] arterial partial pressure of [MASK] dioxide over [MASK] [MASK] oxygen [MASK] sea level. that is to say, at sea [MASK] the arterial partial pressure [MASK] co2 is maintained [MASK] very close to 5. 3 [MASK] [MASK] [MASK] or 40 mmhg ) under a [MASK] [MASK] of circumstances, at the expense of the arterial partial pressure [MASK] o2, which is [MASK] to vary within a very wide range of
pred_top_1:     faster ( hyperpnea ). the exact degree of hyperpnea is[99.1%] determined by the blood gas homeostat, which regulates the partial pressure[50.3%] of oxygen and carbon dioxide in the[92.1%] arterial blood. this homeostat prioritizes the control[10.4%] of the[92.9%] arterial partial pressure of carbon[99.5%] dioxide over oxygen[33.2%] and[19.3%] oxygen at[89.8%] sea level. that is to say, at sea level[93.8%] the arterial partial pressure of[96.7%] co2 is maintained at[42.5%] very close to 5. 3 g[14.0%]hg[60.1%] ([66.5%] or 40 mmhg ) under a wide[80.1%] range[72.2%] of circumstances, at the expense of the arterial partial pressure of[95.0%] o2, which is thought[20.5%] to vary within a very wide range of
pred_top_2:     faster ( hyperpnea ). the exact degree of hyperpnea are[0.5%] determined by the blood gas homeostat, which regulates the partial pressures[14.5%] of oxygen and carbon dioxide in arterial[2.0%] arterial blood. this homeostat prioritizes the regulation[5.4%] of an[2.3%] arterial partial pressure of nitrogen[0.2%] dioxide over carbon[11.4%] of[9.9%] oxygen above[4.4%] sea level. that is to say, at sea levels[4.5%] the arterial partial pressure for[0.9%] co2 is maintained ([21.4%] very close to 5. 3 mm[11.4%]2[5.9%]hg[18.0%] or 40 mmhg ) under a large[3.5%] variety[17.1%] of circumstances, at the expense of the arterial partial pressure,[0.9%] o2, which is known[11.3%] to vary within a very wide range of
pred_top_3:     faster ( hyperpnea ). the exact degree of hyperpnea was[0.4%] determined by the blood gas homeostat, which regulates the partial levels[5.4%] of oxygen and carbon dioxide in an[0.7%] arterial blood. this homeostat prioritizes the determination[4.1%] of high[0.7%] arterial partial pressure of sulfur[0.1%] dioxide over the[7.7%]2[7.8%] oxygen below[1.1%] sea level. that is to say, at sea,[1.4%] the arterial partial pressure in[0.6%] co2 is maintained to[16.6%] very close to 5. 3 μ[7.6%]w[3.6%]2[1.8%] or 40 mmhg ) under a broad[2.7%] number[7.4%] of circumstances, at the expense of the arterial partial pressure for[0.6%] o2, which is expected[8.2%] to vary within a very wide range of
```

# Dataset
The canonical BERT pre-training datasets of BookCorpus and Englins Wikipedia were used.

BooksCorpus  
* A dataset containing over 11,000 unpublished books scraped from the web, covering a variety of genres. It includes approximately 800 million words.
* HuggingFace dataset https://huggingface.co/datasets/bookcorpus/bookcorpus.
  
English Wikipedia
* A cleaned-up version of English Wikipedia articles, consisting of about 2.5 billion words. Only the text content was used (lists, tables, and headers were excluded).
* Huggingface dataset https://huggingface.co/datasets/wikimedia/wikipedia. 
* https://huggingface.co/datasets/legacy-datasets/wikipedia

I counted a total of 4.01B words across both datasets, which is greater than the 3.3B words reported in the [BERT paper](https://arxiv.org/abs/1810.04805)

| Dataset       | # words |
|---------------|---------|
| BookCorpus    | 840.01M |
| Wikipedia(En) | 3.17B   |
| **TOTAL**         | **4.01B**   |


# Tokenization

BERT uses WordPiece, a subword tokenization method, as its tokenizer.
This approach strikes a balance between word-level and character-level tokenization, allowing BERT to efficiently handle a wide vocabulary while managing rare or out-of-vocabulary words.

### Details of WordPiece in BERT:
* How It Works: WordPiece breaks down text into smaller units (subwords or word pieces) based on a pre-trained vocabulary.
It starts with individual characters and iteratively merges them into larger tokens (e.g., "playing" might be split into "play" and "##ing"),
guided by a likelihood-based algorithm that maximizes the probability of the training corpus. The tokenizer is trained on the same dataset as BERT (BooksCorpus and English Wikipedia), ensuring it reflects the statistical properties of the pre-training corpus.
* Vocabulary Size: BERT’s WordPiece tokenizer has a vocabulary of 30,000 tokens, which includes whole words, subwords, and special tokens like [CLS] (for classification) and [SEP] (to separate sentences).
However, the Huggingface's BERT WordPiece tokenizer has vocabulary size of 30,522.
* Special Tokens: 
```
[CLS]: Added at the beginning of every input sequence for classification tasks.
[SEP]: Used to separate sentences in tasks like Next Sentence Prediction or to mark the end of a single sequence.
[PAD]: Used for padding shorter sequences to match the maximum length.
[MASK]: Used during pre-training for the Masked Language Modeling task.
##: A prefix for subword pieces that attach to a previous token (e.g., "playing" → "play" + "##ing").
```

The algorithm used to score which subword pairs to merge is given by

$$score = \frac{\text{freq-of-pair}}{\text{freq-of-first-element}  \times  \text{freq-of-second-element}}$$

By dividing the frequency of the pair by the product of the frequencies of each of its parts,
the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary.

### Why WordPiece?
* Efficiency: It reduces the vocabulary size compared to full-word tokenization, making it computationally manageable.
* Flexibility: It can handle unseen words by breaking them into known subwords (e.g., "unhappiness" → "un" + "##happi" + "##ness").
* Bidirectionality: It pairs well with BERT’s bidirectional architecture, as it preserves meaningful chunks of text for contextual understanding.

### Byte Pair Encoding as an alternative tokenizer to WordPiece
* Byte Pair Encoding (BPE) is a similar tokenization algorithm to WordPiece, and is used in models such as GPT and BART.
* The vocabulary training algorithm for BPE is very similar to WordPiece in that from a set of individual characters,
the vocabulary is constructed by iteratively merging pairs of subword units. However, the subword pairs are merged according
to the frequency at which they appear in the training corpus. Subword pairs that most commonly appear next to eachother are merged first.
* The BPE vocabulary can be constructed from the ASCII and Unicode characters, or from the byte-level representations.

### Cased vs. Uncased
This implementation of BERT uses the uncased tokenizer.
* Uncased means that the text has been lowercased before WordPiece tokenization, e.g., John Smith becomes john smith. The Uncased model also strips out any accent markers.
* Cased means that the true case and accent markers are preserved.

Typically, the Uncased model is better unless you know that case information is important for your task (e.g., Named Entity Recognition or Part-of-Speech tagging).

### Tokenization steps
1. Normalization (any cleanup of the text that is deemed necessary, such as removing spaces or accents, Unicode normalization, etc.)
2. Pre-tokenization (splitting the input into words)
3. Running the input through the model (using the pre-tokenized words to produce a sequence of tokens)
4. Post-processing (adding the special tokens of the tokenizer, generating the attention mask and token type IDs)

### Training the Tokenizer
I experimented with training the tokenizer from scratch [see 'train_tokenizer.py'](bert/train_tokenizer.py),
but ultimately just used the pre-trained Huggingface tokenizers for BERT.

### Huggingface Tokenizer

[BertTokenizer](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertTokenizer)
* BERT tokenizer based on WordPiece
* [github implementation](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/bert/tokenization_bert.py)
[BertTokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/bert#transformers.BertTokenizerFast)
* "fast" BERT tokenizer based on WordPiece

## Dataset Tokenization and Fixed Sequence Length
BERT requires data input minibatches of fixed sequence length. In the paper, the authors report that the BERT model is
pre-trained first on sequences of length 128 for 90% of the steps, and then on sequence of length 512 for the
remaining 10% of the steps.
A shorter length input sequence limits the range of tokens that the Transformer can attend to when predicting a masked token.

I experimented with the suggested training procedure, but quickly realized that training on 512-length sequences
required considerably more GPU vRAM and iterations to complete a training epoch due to the reduced minibatch size.
Therefore, I selected pre-train the BERT model only on 128-length sequences for the 1M iteration training experiment described above.

### Mini-batch construction
To construct a dataset minibatch, each variable length text sequence in the training corpus must be truncated to fixed token length (128)
and then padded with the `[PAD]` token.
A attention mask {0,1} is prepared to mark the tokens that shoudl be attended to or ignored by the attention layers of the model.
The `[PAD]` tokens should be ignored.

## Whole Word Masking procedure
The BERT Masked Language Model pre-training procedure calls for 15% of the tokens in the input sequence to be randomly masked.
However, of those selected tokens, 80% ared replaced with a [MASK] token, 10% are replaced with a random token, and the
remaining 10% are left unchanged.

I follow the example from [the official BERT implementation](https://github.com/google-research/bert/blob/master/README.md) and apply Whole Word Masking, where we always mask _all_
of the tokens correstponding ot a word at once.

In the following example, the Whole Word Masking procedure ensures that the word 'philammon' is masked in it's entirety.
```
Input Text:              the man jumped up , put his basket on phil ##am ##mon ' s head
Original Masked Input:   [MASK] man [MASK] up , put his [MASK] on phil [MASK] ##mon ' s head
Whole Word Masked Input: the man [MASK] up , put his basket on [MASK] [MASK] [MASK] ' s head
```

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

I selected to train the BERT model using Pre-LayerNormalization, where the layer norm is applied to the input of each self-attention 
and feed forward block rather than to the output after the residual connection.
The BERT model trained above actually used the "Pre-LayerNormalization" approach described in described in ["On Layer Normalization in the Transformer Architecture" by Xiong et al. (2020)](https://arxiv.org/abs/2002.04745). The Layer Norm is applied to the input of each self-attention and feed forward block rather than to the output after the residual connection.
Pre-LayerNormalization allows for the use of a greater learning rate learning rate from 3e-4 (rather than 1e-4), and reduces the necessity for a warm-up stage.
According to the authors of the paper using such a high learning rate for the standard post-LN BERT would lead to optimization divergence.

Due to GPU vRAM limitations, I trained with a mini-batch size of 128. I used 2x gradient accumulation to achieve an effective batch size of 256,
which matches the batch size used to pre-train the official BERT model. It's worth noting that using 2x gradient accumulation effectively halfed
the number of gradient updates experienced at training time. The correct resolution to this would be to double the training period.

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

## Evaluation Metrics

### Perplexity

Perplexity is the exponentiation of the average negative log-likelihood (NLL) of the correct (masked) tokens, calculated over a dataset.
For a masked language model like BERT, it quantifies how "surprised" the model is when predicting the masked tokens — lower perplexity indicates better predictions, meaning the model assigns higher probabilities to the correct tokens.
Mathematically, for a dataset with N masked tokens, perplexity (PPL) is defined as:
$$
PPL = \exp⁡(-\frac{1}{N}\sum_{i=1}^{N}\log⁡P(w_i∣w_{context}))
$$

Where:
* $w_i$: The i-th masked token (the ground truth).
* $w_{context}$: The surrounding context (unmasked tokens) used to predict $w_i$.
* $P(w_i∣w_{context})$: The model’s predicted probability of the correct token $w_i$ given the context.
* N: The total number of masked tokens in the evaluation dataset.


BERT is trained with a masked language modeling objective, where some tokens in a sentence are replaced with [MASK], and the model predicts the original tokens based on the bidirectional context.

Perplexity evaluates this task:

1. Input Preparation:
* A sentence is tokenized, and a fraction of tokens (e.g., 15% in BERT’s pretraining) are masked.
* Example: "The cat sat on the [MASK]." → Predict "mat".

2. Model Prediction:
* BERT outputs a probability distribution over its vocabulary for each [MASK] position.
* For the correct token (e.g., "mat"), the model assigns a probability P("mat"∣"The cat sat on the [MASK]").

3. Loss Calculation:
* The negative log-likelihood is computed: $−\log⁡P(\text{"mat"}∣\text{context})$
* This measures how confident the model is—low NLL means high probability for the correct token.

4. Perplexity:
* Average the NLL across all masked tokens in the dataset, then exponentiate to get perplexity.
* Example: If the average NLL is 2.3, then $PPL=e^{2.3} \approx 10$, meaning the model is effectively "choosing" between 10 equally likely options on average.


Low Perplexity: The model predicts masked tokens with high confidence (e.g., PPL=2 means it’s like choosing between 2 options).

High Perplexity: The model is uncertain, assigning low probabilities to the correct tokens (e.g., PPL=100 means it’s like choosing among 100 options).


# References
* BERT paper https://arxiv.org/abs/1810.04805
* Blog post https://jalammar.github.io/illustrated-bert/
* Google Research BERT implementation https://github.com/google-research/bert
* Huggingface BERT model https://huggingface.co/google-bert/bert-base-uncased
* Huggingface https://huggingface.co/docs/transformers/model_doc/bert
* Huggingface language model training examples https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling
* Huggingface tokenization tutorial https://huggingface.co/learn/nlp-course/en/chapter6/1?fw=pt
