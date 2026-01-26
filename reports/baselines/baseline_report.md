# Baseline Results (Binary: Attack vs Benign)

- train: 1433536 rows
- val:   363450 rows
- test:  516824 rows

## LogReg

### Validation
- macro F1: 0.4994
- attack recall: 0.0028
Confusion matrix (rows=true, cols=pred):
```
[[360031   1240]
 [  2173      6]]
```
Classification report:
```
              precision    recall  f1-score   support

           0     0.9940    0.9966    0.9953    361271
           1     0.0048    0.0028    0.0035      2179

    accuracy                         0.9906    363450
   macro avg     0.4994    0.4997    0.4994    363450
weighted avg     0.9881    0.9906    0.9893    363450

```
### Test
- macro F1: 0.8507
- attack recall: 0.6207
Confusion matrix (rows=true, cols=pred):
```
[[384620    797]
 [ 49843  81564]]
```
Classification report:
```
              precision    recall  f1-score   support

           0     0.8853    0.9979    0.9382    385417
           1     0.9903    0.6207    0.7631    131407

    accuracy                         0.9020    516824
   macro avg     0.9378    0.8093    0.8507    516824
weighted avg     0.9120    0.9020    0.8937    516824

```
## RandomForest

### Validation
- macro F1: 0.5451
- attack recall: 0.0542
Confusion matrix (rows=true, cols=pred):
```
[[361039    232]
 [  2061    118]]
```
Classification report:
```
              precision    recall  f1-score   support

           0     0.9943    0.9994    0.9968    361271
           1     0.3371    0.0542    0.0933      2179

    accuracy                         0.9937    363450
   macro avg     0.6657    0.5268    0.5451    363450
weighted avg     0.9904    0.9937    0.9914    363450

```
### Test
- macro F1: 0.5568
- attack recall: 0.1378
Confusion matrix (rows=true, cols=pred):
```
[[385284    133]
 [113299  18108]]
```
Classification report:
```
              precision    recall  f1-score   support

           0     0.7728    0.9997    0.8717    385417
           1     0.9927    0.1378    0.2420    131407

    accuracy                         0.7805    516824
   macro avg     0.8827    0.5687    0.5568    516824
weighted avg     0.8287    0.7805    0.7116    516824

```
