# QuoteCSE: detecting Contextomized quotes in News Headlines

## Task
Given a news title quote *T* and body quote *B*, we want to detect contextomized quote in news headlines.
So, we present QuoteCSE, a contrastive learning framework that represents the embedding of news quotes.

## Data
We prepared two datasets.
1. To pretrain
We construct 86,275 datasets consisting of title quotes, positive sentences, and negative sentences to train QuoteCSE model. Due to data size issues, only 200 samples were uploaded to this repository. 
2. To downstream task detecting the contextomized quotes
we construct 1,600 datasets consisting of *T*, *B*, and labels to fine-tune the trained QuoteCSE model.

We think that this dataset can also be uitilzed for other tasks to detect contextomized quotes.


## Model
We present QuoteCSE, a contrastive learning framework that represents the embedding of news quotes. In addition, we implemented a classifier to classify whether the title quote *T* is contextomized or not using embedding obtained by QuoteCSE. It can be executed using the command below.


## Example Usage
- Training QuoteCSE model with Contrastive Learning
```python
python train.py 
```

- Detecting contextomized quote to use QuoteCSE model
```python
python contextomized_quote_detection.py 
```


## Reference

