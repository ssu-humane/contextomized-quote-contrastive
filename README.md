# QuoteCSE: detecting Contextomized quotes in News Headlines

## Task
Given a news title quote *T* and body quote *B*, we want to detect contextomized quote in news headlines.

## Data
We prepared two datasets.
1. To pre-train
We construct 86,275 datasets consisting of title quotes, positive sentences, and negative sentences to train QuoteCSE model. Due to data size issues, only 200 samples were uploaded to this repository. 
2. To downstream task detecting the contextomized quotes
we construct 1,600 datasets consisting of *T*, *B*, and labels to fine-tune the trained QuoteCSE model.
Label 1 means 'Contextomized', and label 0 means 'Modified'. The table below is an example for each label.

|title quote|body quotes|label|
|------|---|---|
|"A debt crisis, like Greece, is on the horizon"|"If we do not maintain our fiscal health, we may end up like Greece" <br/> "Wasted budgets should be reallocated to areas in need through the reconstruction of public expenditure"|Contextomized|
|"Avoid unnecessary gatherings altogether"|"Since being vaccinated, I had planned to travel somewhere in the summer, but..." <br/> "Is is not a matter of prohibiting or permitting specific activities, but of avoiding unnecessary gatherings, appointments, and going out altogether..."|Modified|

We think that this dataset can also be uitilized for other tasks to detect contextomized quotes.


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

