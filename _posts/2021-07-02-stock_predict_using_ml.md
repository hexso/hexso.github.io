---
layout: post
title: 머신러닝을 이용한 주식 예측(1)
date: 2021-07-02 01:23:18 +0800
last_modified_at: 2021-07-02 01:23:18 +0800
tags: [Machine learning]
toc:  true
---


회사에서 Pytorch를 이용한 Machine Learning관련 교육을 들었다.  

간단하게 이를 이용하여 구현해보고자 결정

좀 더 흥미를 가지고 할만한 주제를 고민 후, 주식을 이에 활용해보면 어떨까 생각.

게다가 기존에 구글링하면 자료들이 많아서 쉽게 접근하기 쉽다고 생각했다. 

[predict_stock 깃허브 ](https://github.com/hexso/predict_stock)





# LSTM

LSTM은 RNN(Recurrent Neural Networks)의 한 종류로써, 순서가 중요한 요소일 때 흔히 적용하는 RNN의 한 종류다. 문장과 같은 단어가 문장 안에서의 순서가 중요한 경우나, 주가와 같은 시계열 데이터셋에서 효과적인 모델이다.

![The repeating module in an LSTM contains four interacting layers](https://t1.daumcdn.net/cfile/tistory/999F603E5ACB86A005)

RNN과의 큰 차이점중 하나는 Cell state가 있다는 것.  

내가 이해한 바로는 이 Cell State에 의해 기존에 그 전에 들어온 Data들로 학습(?)되고 나중 Data를 이용해 Learning을 하는데 도움이 된다.



### 가정

주식에서 기술적 분석이라는 방법이 있다. 오직 차트같은 지표들만을 이용하여 트레이드 하는 방식. 

따라서 오직 이러한 Data들만을 토대로 미래 주식가격을 예측할 수 있다고 가정한다.
