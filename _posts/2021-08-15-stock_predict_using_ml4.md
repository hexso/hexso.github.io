---
layout: post
title: 머신러닝을 이용한 주식 예측(4)
date: 2021-08-15 01:23:18 +0800
last_modified_at: 2021-08-15 01:23:18 +0800
tags: [Machine learning]
toc:  true 
---

LSTM의 경우 Data가 자꾸 전 데이터를 따라가려는 성질이 많은것 같다.

그래서 우선 머신러닝에 대해 기초를 배울겸 Linear구조로 구현하여 다음날에 대한 예측을 해보았다.

Layer의 구조는 아래와 같다.



```python
def __init__()
    self.layer_1 = nn.Linear(self.input_size, 64)
    self.layer_2 = nn.Linear(64, 64)
    self.layer_out = nn.Linear(64, 1)

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.1)
    self.batchnorm1 = nn.BatchNorm1d(64)
    self.batchnorm2 = nn.BatchNorm1d(64)
    self.loss_function = nn.BCEWithLogitsLoss()
    self.optimizer = optim.Adam

def forward(inputs):
    x = self.relu(self.layer_1(inputs))
    x = self.batchnorm1(x)
    x = self.relu(self.layer_2(x))
    x = self.batchnorm2(x)
    x = self.dropout(x)
    x = self.layer_out(x)
    return x
```



### 내일 주식 상승 하락 예측

내일의 주식가격이 상승할지 하락할지를 예측하도록 해보았다.

Data는 삼성전자 주식을 1997년 7월 처음 상장했을 때부터 지금까지 데이터를 사용하였다.

학습데이터는 58%정도의 정확도를 보여주었다.

Epoch 050: | Loss: 0.66487 | Acc: 58.952





밑의 표는 sklearn에 있는  confusion_matrix함수를 실행한 결과다.

예측하고싶은 데이터에 대한 실제 결과와의 차이를 표로 쉽게 알 수 있다.

|           | 예측 하락 | 예측 상승 |
| --------- | --------- | --------- |
| 실제 하락 | 414       | 524       |
| 실제 상승 | 415       | 627       |



sklearn의 classification_report에 대한 결과.

train data 와 test data가 비슷한 것을 알 수 있다.

다만 정확도가 좀 떨어진다.

                      precision    recall  f1-score   support
    
        (하락)0.0       0.50      0.44      0.47       938
        (상승)1.0       0.54      0.60      0.57      1042
    
        accuracy                           0.53      1980
       macro avg       0.52      0.52      0.52      1980
    weighted avg       0.52      0.53      0.52      1980






### 내일 주식이 3%이상 상승할지 예측

이번에는 주식이 3%이상 오를지를 예측하도록 해보았다.

train Data는 예상이상의 정확도를 보여주었다.

Epoch 010: | Loss: 0.60555 | Acc: 68.500
Epoch 020: | Loss: 0.48656 | Acc: 86.500
Epoch 030: | Loss: 0.42147 | Acc: 87.750
Epoch 040: | Loss: 0.31748 | Acc: 95.250
Epoch 050: | Loss: 0.24089 | Acc: 95.250



그러나 test Data에서는 전혀 반대의 결과가 나왔다.

confusion_matrix함수를 실행한 결과다.

|             | 3%이상 안오름 | 3%이상 오름 |
| ----------- | ------------- | ----------- |
| 실제 안오름 | 2977          | 390         |
| 실제 오름   | 564           | 89          |



classification_report에 대한 결과다.

```
              	precision    recall  f1-score   support

         0.0       0.84      0.88      0.86      3367
         1.0       0.19      0.14      0.16       653

    accuracy                           0.76      4020
   macro avg       0.51      0.51      0.51      4020
weighted avg       0.73      0.76      0.75      4020
```

아주 낮은 정확도를 나타냈다. overfiting은 아닌것 같은게.. 하나의 주식 Data만을 학습시킨게 아니라, 

여러개의 다른 주식들의 data를 2020-06-01부터 지금까지의 Data를 학습시켰고, 또 다른 주식으로 Predict했을때의 결과다.

이 때 다른 주식들을 계속 학습 시켰을 때는 정확도가 꾸준히 높은 정확도를 나타냈었다.. 

따라서 왠지는 아직 잘 모르겠다.. 분석 필요.

