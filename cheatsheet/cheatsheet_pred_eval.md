## Prediction & Evaluation

Related to the evaluation.

#### Evaluation
```python
# Training
model.train()

# Evaluation
model.eval()
with torch.no_grad():
    # code
    # code
torch.cuda.empty_cache()
```

#### Accuracy
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(trues,preds)
# The same as {precision_score, recall_score,f1_score}
```

#### Transfer logits to labels
```python
logits = outputs.logits # with shape (batch_size, label_number),
probs = logits.softmax(dim=-1).data.cpu() # logits to probs
_, pred = torch.max(probs, 1) # probs to label ids.
```