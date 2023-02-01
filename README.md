# A Contrastive Learning Framework for Detecting Contextomized News Quotes (QuoteCSE)

## Task
Given a news title quote *T* and body quote *B*, we want to detect contextomized quote in news headlines.

## Data

### Pretraining corpus

```
data/pretrain_data_sample.pkl
```
We present a sampled dataset of unlabeled corpus used for the QuoteCSE pretraining. Each data instance consists of title quote, positive sample, and negative sample. The positive and negative samples were selected by SentenceBERT, and the assignments are updated during training.

### Manually annotated dataDownstream task to detect the contextomized quotes
<br/>We construct 1,600 datasets consisting of *T*, *B*, and labels to fine-tune the trained QuoteCSE model. Label 1 means 'Contextomized', and label 0 means 'Modified'.<br/>
Contextomized quote is a title quote that does not semantically match any quotes in the body text and distorts the speaker's intention.
Modified quote is a title quote that is different in form but semantically matches quotes in the body text.

<br/> This table is an example for each label.
|title quote|body quotes|label|
|------|---|:---:|
|"A debt crisis, like Greece, is on the horizon"|"If we do not maintain our fiscal health, we may end up like Greece" <br/> "Wasted budgets should be reallocated to areas in need through the reconstruction of public expenditure"|Contextomized|
|"Avoid unnecessary gatherings altogether"|"Since being vaccinated, I had planned to travel somewhere in the summer, but..." <br/> "It is not a matter of prohibiting or permitting specific activities, but of avoiding unnecessary gatherings, appointments, and going out altogether..."|Modified|

We think that this dataset can also be uitilized for other tasks to detect contextomized quotes.


## Model
We present QuoteCSE, a contrastive learning framework that represents the embedding of news quotes. In addition, we implemented a classifier to classify whether the title quote *T* is contextomized or not using embedding obtained by QuoteCSE. It can be executed using the command below.


## Example Usage
- QuoteCSE pretraining
```python
python train.py 
```

- Detecting contextomized quote to use QuoteCSE model
```python
python contextomized_quote_detection.py 
```


## Reference

