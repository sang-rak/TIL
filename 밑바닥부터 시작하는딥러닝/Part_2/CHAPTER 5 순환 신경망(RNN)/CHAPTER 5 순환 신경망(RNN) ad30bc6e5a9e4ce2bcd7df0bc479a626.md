# CHAPTER 5 순환 신경망(RNN)

# 1. 확률과 언어 모델

피드포워드 신경망

- 흐름이 단방향
- 시계열 데이터의 성질(패턴)을 충분히 학습할 수 없음

순환 신경망의 등장

## 1.1 word2vec을 확률 관점에서 바라보다

CBOW 모델

- CBOW 모델의 학습 → 손실 함수(말뭉치 전체의 손실 함수의 총합)를 최소화하는 가중치 매개 변수를 찾는다 → 맥락으로부터 타깃을 정확하게 추측 가능
- 맥락 안의 단어 순서가 무시된다는 한계가 있음

    말뭉치: w1, w2, w3, ..., wt

    맥락을 좌우 대칭으로 생각

    t번째 단어를 타깃으로 그 전후 단어(t-1)번쨰 단어와 (t+1)번째 단어를 맥락으로 취급

    ![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled.png)

    w(t-1)과 w(t+1)이 주어졌을 때 타깃이 wt가 될 확률을 수식으로 표현하면

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%201.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%201.png)

- 맥락을 왼쪽 윈도우 만으로 한정

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%202.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%202.png)

w(t-2)과 w(t-1)이 주어졌을 때 타깃이 wt가 될 확률(CBOW 모델이 출력할 확률)을 수실으로 표현하면

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%203.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%203.png)

CBOW 모델이 다루는 손실함수

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%204.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%204.png)

## 1.2 언어 모델

단어 나열에 확률을 부여

특정 단어의 시퀀스에 대해서 그 시퀀스가 일어날 가능성이 어느 정도인지(얼마나 자연스러운 단어 순서인지)를 확률로 평가한다.

- 언어 모델의 사용
    1. 기계 번역과 음성 인식
    2. 새로운 문장을 생성

w1, ... , wm 이라는 m개 단어로 된 문장이 있을 때

w1, ... , wm 순서로 출현할 확률 P(w1, ..., wm)

여러 사건이 동시에 일어날 확률이므로 동시  확률이라고 한다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%205.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%205.png)

동시확률은 사후 확률의 총곱으로 나타낼 수 있다.

이 사후 확률은 타깃 단어보다 왼쪽에 있는 단어를 맥락(조건)으로 했을때의 확률이라는 것에 주목해야 한다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%206.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%206.png)

확률의 곱셈정리

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%207.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%207.png)

A, B가 모두 일어날 확률 P(A, B)는 B가 일어날 확률 P(B)와 B가 일어난 후 A가 일어날 확률 P(A|B)를 곱한 값과 같다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%208.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%208.png)

## 1.3 CBOW 모델을 언어 모델로?

- word2vec의 CBOW모델을 언어 모델에 적용하려면 맥락의 크기를 특정 값으로 한정하여 근사적으로 나타낼 수 있다.
- 맥락의 크기는 임의 길이로 설정할 수 있지만 결국 특정 길이로 '고정'된다.

    - -예를 들어 왼쪽 10개의 단어를 맥락으로 CBOW 모델을 만든하고 하면 그 맥락보다 더 왼쪽에 있는 단어의 정보는 무시 된다.

- CBOW 모델의 맥락 크기를 키울 수는 있으나 맥락 안의 단어 순서가 무시된다는 한계가 있다.
- 맥락의 단어 순서를 고려하기 위해 맥락의 단어 벤터를 은닉층에서 연결하는 방식을 생각할 수 있으나 맥락의 크기에 비례해 가중치 매개변수가 늘어난다는 문제가 발생한다.

- 그래서 순환신경망 즉 RNN이 등장하게 되었는데 RNN은 맥락이 아무리 길더라도 맥락의 정보를 기억하는 메커니즘을 갖추고 있기에 아무리 긴 시계열 데이터에도 대응 할 수 있다.

# 2. RNN이란

## 2.1 순환하는 신경망

순환하기 위해서는 닫힌 경로가 필요하다.

닫힌 경로 혹은 순환하는 경로가 존재해야 단어가 같은 장소를 반복해 왕래할 수 있고 데이터가 순환하면서 과거의 정보를 기억하는 동시에 최신 데이터로 갱신 될 수 있다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%209.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%209.png)

t: 시각

시계열 데이터 (x0, x1, ..., xt, ...)가 RNN계층에 입력되고 이에 대응해(h0, h1, ..., ht, ...)가 출력된다.

각 시각에 입력되는 xt를 벡터라고 가정했을 때

문장(단어 순서)을 다루는 경우를 예로 든다면 각 단어의 분산 표현(단어 벡터)이 xt가 되며 이 분산 표현이 순서대로 하나씩 RNN계층에 입력된다.

## 2.2 순환 구조 펼치기

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2010.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2010.png)

RNN계층의 순환 구조를 펼침으로써 오른쪽으로 성장하는 긴 신경망으로 변신

피드포워드 신경망(데이터가 한 방향으로만 흐른다)과 같은 구조이지만 위그림에서는 다수의 RNN계층 모두가 실제로는 '같은 계층' 인 것이 지금까지의 신경망과는 다른 점이다.

각 시각의 RNN계층은 그 계층으로의 입력과 1개 전의 RNN 계층으로부터의 출력을 받는데 이 두 정보를 바탕으로 현 시각의 출력을 계산한다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2011.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2011.png)

Wx: 입력 x를 출력 h로 변환하기 위한 가중치

Wh: 1개의 RNN출력을 다음 시각의 출력으로 변환하기 위한 가중치

b: 편향

h(t-1), xt:행백터

ht는 다른 계층을 향해 위쪽으로 출력 되는 동시에 다음 시각의 RNN(자기 자신)을 향해 오른쪽으로도 출력된다. RNN의 출력 ht는 은닉상태 혹은 은닉상태 벡터라고 한다.

RNN은 h라는 '상태'를 가지고 있으며 위의 식을 형태로 갱신 된다고 해석 할 수 있다.

RNN계층을 '상태를 가지는 계층' 혹은 메모리가 있는 계층 이라고 한다.

## 2.3 BPTT

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2012.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2012.png)

순환 구조를 펼친후의 RNN에는 오차역전파법을 적용할 수 있다.

먼저 순전파를 수행하고 이어서 역전파를 수행하여 원하는 기울기를 구할 수 있다.

여기서 오차역전파법은 ' 시간방향으로 펼친 신경망의 오차역전파법'이란 뜻으로 BPTT(Backpropagation Through Time)라고 한다.

문제점

- 시계열 데이터의 시간 크기가 커지는 것에 비례하여 BPTT가 소비하는 컴퓨팅 자원도 증가
- 시간 크기가 커지면 역전파 시의 기울기가 불안정해짐

## 2.4 Truncated BPTT

Truncated BPTT: 시간축 방향으로 너무 길어진 신경망을 적당한 시점에서 잘라내어 작은 신경망 여러 개로 만들어 잘라낸 작은 신경망에서 오차역전파법을 수행한다.

- 계층이 너무 길면 계산량과 메모리 사용량 등이 문제가 되고 계층이 길어짐에 따라 신경망을 하나 통과할 때마다 기울기 값이 조금씩 작아져서 이전 시각 t까지 역전파되기 전에 0이되어 소멸할 수도 있다.
- 순전파의 연결을 그대로 유지하면서(데이터를 순서대로 입력해야 한다) 역전파의 연결은 적당한 길이로 잘라내 잘라낸 신경망 단위로 학습을 수행한다.
- 역전파의 연결을 잘라버리면 그보다 미래의 데이터에 대해서는 생각할 필요가 없어지기 때문에 각각의 블록 단위로 미래의 블록과는 독립적으로 오차역전파법을 완결 시킨다.
    - 블록: 역전파가 연결되는 일련의 RNN게층

- 순전파를 수행하고 그 다음 역전파를 수행하여 원하는 기울기를 구한다.
- 다음 역전파를 수행할 때 앞 블록의 마지막 은닉 상태인 ht가 필요하다.

    ht로 순전파가 계속 연결 될 수 있다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2013.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2013.png)

## 2.5 Truncated BPTT의 미니배치 학습

미니배치 학습을 수행할 때는 각 미니 배치의 시작 위치를 오프셋으로 옮겨준 후 순서대로 제공하면 된다. 또한 데이터를 순서대로 입력하다가 끝에 도달하면 다시 처음부터 입력하도록 한다.

- 오프셋

    일반적으로 동일 오브젝트 안에서 오브젝트 처음부터 주어진 요소나 지점까지의 변위차를 나타내는 정수형 

    예) 문자 A의 배열이 abcdef를 포함한다면 'c' 문자는 'A'시작점에서 2의 오프셋을 지닌다고 할 수 있다.

미니배치를 두 개로 구성해 학습할 때

첫 번째 미니배치 때는 처음부터 순서대로 데이터를 제공

두 번째 미니배치 때는 500번째 데이터를 시작 위치로 정하고 그 위치부터 다시 순서대로 데이터를 제공

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2014.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2014.png)

→Truncated BPTT의 원리는 단순하지만 데이터 제공 방법 두 가지를 주의

: 데이터를 순서대로 제공하기, 미니배치별로 데이터를 제공하는 시작 위치를 옮기기

# 3. RNN 구현

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2015.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2015.png)

길이가 T인 시계열 데이터를 받는다.

각 시각의 은닉 상태를 T개 출력한다.

모듈화를 생각해 위의 그림의 신경망을 '하나의 계층'으로 구현한다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2016.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2016.png)

xs를 입력하면 hs를 출력하는 단일 계층

Time RNN계층 내에서 한 단계의 작업을 수행하는 계층을 'RNN계층'이라 하고

T개 단계분의 작업을 한꺼번에 처리하는 계층을 'Time RNN계층'이라 한다.

## 3.1 RNN 계층 구현

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2017.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2017.png)

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2018.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2018.png)

N: 미니배치 크기 D: 입력 벡터의 차원 수 H: 은닉 상태 벡터의 차원 수

**RNN 처리를 한 단계만 수행하는 RNN클래스 구현**

```python
#RNN 클래스의 초기화와 순전파 메서드를 구현
class RNN:
  def __init__(self, Wx, Wh, b):
    self.params = [Wx, Wh, b]
    #가중치 2개, 편향 1개
    self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    #각 매개변수에 대응하는 형태로 기울기를 초기화한 후 grads에 저장
    self.cache = None
    #역전파 계산 시 사용하는 중간 데이터를 담은 cache를 None으로 초기화
    
  def forward(self, x, h_prev):
    #아래로부터의 입력 x와 왼쪽으로부터의 입력 h_prev
    Wx, Wh, b = self.params
    t = np.matmul(h_prev, Wh) + np.matmul(x, Wx) + b
    #matmul: 행렬의 곱
    h_next = np.tanh(t)
    
    self.cache = (x, h_prev, h_next)
    return h_next
```

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2019.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2019.png)

```python
#RNN계층의 역전파 메서드 구현
def backward(self, dh_next):
  Wx, Wh, b = self.params
  x, h_prev, h_next = self.cache

  dt = dh_next*(1-h_next**2)
  db = np.sum(dt, axis=0)
  dWh = np.matmul(h_prev.T, dt)
  dh_prev = np.matmul(dt, Wh.T)
  dWx = np.matmul(x.T, dt)
  dx = np.matmul(dt, Wx.T)

  self.grads[0][...] = dWx
  self.grads[1][...] = dWh
  self.grads[2][...] = db

  return dx, dh_prev
```

## 3.2 Time RNN 계층 구현

RNN계층 T개를 연결한 신경망 Time RNN계층

RNN 계층의 은닉 상태 h를 인스턴스 변수로 유지 해 은닉 상태를 '인계'받는 용도로 이용한다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2020.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2020.png)

RNN계층의 은닉 상태를 Time RNN계층에서 관리하면 Time RNN사용자는 RNN계층 사이에서 은닉 상태를 '인계하는 작업'을 생각하지 않아도 된다.

```python
#RNN클래스를 이용해 T개 단계의 처리를 한꺼번에 수행하는 계층을 
#TimeRNN이란 이름의 클래스로 완성한다.
class TimeRNN:
  def __init__(self, Wx, Wh, b, stateful=false):
    #초기화 메서드는 가중치, 편향, stateful이라는 boolean값을 인수로 받음
    #stateful=True : Time RNN계층이 은닉 상태를 유지한다.->아무리 긴 시계열 데이터라도 Time RNN계층의 순전파를 끊지 않고 전파한다.
    #stateful=False: Time RNN 계층은 은닉 상태를 '영행렬'로 초기화한다.상태가 없다.
    self.params = [Wx, Wh, b]
    self.grads = [np.zeors_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
    self.layers = None
    #layers : 다수의 RNN계층을 리스트로 저장하는 용도
    
    self.h, self.dh = None, None
    #h: forward() 메서드를 불렀을 때 마지막 RNN 계층의 은닉 상태를 저장
    #dh: backward()를 불렀을 때 하나 앞 블록의 은닉 상태의 기울기를 저장한다.
    self.stateful = stateful
    
  def set_state(self, h):
    #Time RNN계층의 은닉 상태를 설정하는 메서드
    self.h = h
    
  def reset_state(self):
    #은닉 상태를 초기화하는 메서드
    self.h = None

#순전파 구현
def forward(self, xs):
  #아래로부터 입력 xs(T개 분량의 시계열 데이터를 하나로 모은 것)를 받는다.
  Wx, Wh, b = self.params
  N, T, D = xs.shape #N: 미니배치 크기 D: 입력 벡터의 차원 수
  D, H = Wx.shape
  
  self.layers = []
  hs = np.empty((N, T, H), dtype='f')
  #출력값을 담을 그릇 hs를 준비한다.
  
  if not self.stateful or self.h is None:
    self.h = np.zeros((N, H), dtype='f')
    #h: RNN 계층의 은닉 상태. 
    #self.h=None: 처음 호출 시에는 원소가 모두 0인 영행렬로 초기화됨.
    #stateful=False: 항상 영행렬로 초기화
    
  for t in range(T):
    layer = RNN(*self.params)
    # *: 리스트의 원소들을 추출하여 메서드의 인수로 전달
    #self.params에 들어 있는 Wx, Wh, b를 추출하여 RNN 클래스의 __init__()메서드에 전달
    #RNN계층을 생성하여 인스턴스 변수 layers에 추가한다.
    self.h = layer.forward(xs[:, t, :], self.h)
    hs[:, t, :] = self.h
    self.layers.append(layer)
    
  return hs

#역전파 구현
def backward(self, dhs):
  Wx, Wh, b = self.params
  N, T, H = dhs.shape
  D, H = Wx.shape
  
  dxs = np.empty((N, T, D), dtype='f')
  dh = 0
  grads = [0, 0, 0]
  for t in reversed(range(T)):
    layer = self.layers[t]
    dx, dh = layer.backward(dhs[:, t, :] + dh) #합산된 기울기
    #RNN계층의 순전파에서는 출력이 2개로 분기되는데 역전파에서는 
    #각 기울기가 합산되어 전해진다.
    dxs[:, t, :] = dx
    
    for i, grad in enumerate(layer.grads):
      grads[i] +=grad
      
  for i, grad in enumerate(grads):
    self.grads[i][...] = grad
  self.dh = dh
  
  return dxs
```

# 4. 시계열 데이터 처리 계층 구현

## 4.1 RNNLM의 전체 그림

RNNLM(RNN Language Model): RNN을 사용한 언어 모델

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2021.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2021.png)

Embedding: 단어 ID를 단어의 분산 표현(단어 벡터)으로 변환

RNN계층: 은닉 상태를 다음 층으로(위쪽으로) 출력함과 동시에 다음 시각의 RNN 계층으로(오른쪽으로) 출력한다.

RNN계층이 위로 출력한 은닉 상태는 Affine 계층을 거쳐 Softmax계층으로 전해진다.

RNNLM은 지금까지 입력된 단어를 '기억'하고 그것을 바탕으로 다음에 출현할 단어를 예측한다. RNN계층이 과거에서 현재로 데이터를 계속 흘려보내줌으로써 과거의 정보를 인코딩해 저장(기억) 할 수 있다.

## 4.2 Time 계층 구현

시계열 데이터를 한꺼번에 처리하는 계층

Time Embedding, Time Affine...

Time 계층은 간단하게 구현할 수 있다.

예를 들어 Time Affine계층은 Affine계층을 T개 준비해서 각 시각의 데이터를 개별적으로 처리하면 된다.

시계열 버전의 Softmax계층을 구현할 때는 손실 오차를 구하는 Cross Entropy Error 계층도 함께 구현한다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2022.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2022.png)

x0, x1,... : 아래층에서부터 전해지는 점수(확률로 정규화되기 전의 값)

t0, t1,... : 정답 레이블

T개의 Softmax with Loss 계층 각각이 손실을 산출하고 그 손실들을 합산해 평균한 값이 최종 손실이 된다.

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2023.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2023.png)

Time Softmax with Loss 계층도 시계열에 대한 평균을 구하는 것으로 데이터 1개당 평균 손실을 구해 최종 출력으로 내보낸다.

# 5. RNNLM 학습과 평가

## 5.1 RNNLM 구현

```python
import sys
sys.path.append('..')
import numpy as np
from common.time_layers import * 

#SimpleRnnlm클래스는 4개의 Time계층을 쌓은 신경망이다.
class SimpleRnnlm:
    def __init__(self, vocab_sizse, wordvec_size, hidden_size):
      V, D, H = vocab_size, wordvec_size, hidden_size
      rn = np.random.randn

      #가중치 초기화
      embed_W = (rn(V, D) / 100).astype('f)
      rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
      rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
      rnn_b = np.zeros(H).astype('f')
      affine_W = (rn(H, V) / np.sqrt(H)), astype('f')
      affine_b = np.zeros(V).astype('f')
      #RNN 계층과 Affine계층에서 'Xabier초깃값'을 이용

     #계층 생성
      self.layers = [
        TimeEmbedding(embed_W),
        TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
        #Truncated BPTT로 학습한다고 가정하여 Time RNN계층의 stateful=True로 설정 -> TimeRNN계층은 이전 시간의 은닉 상태를 계승할 수 있다.
        TimeAffine(affine_W, affine_b)
      ]
      self.loss_layer = TimeSoftmaxWithLoss()
      self.rnn_layer = self.layers[1]

      #모든 가중치와 기울기를 리스트에 모은다.
      self.params, self.grads = [],[]
      for layer in self.layers:
        self.params += layer.params
        self.grads += layer.grads
```

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2024.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2024.png)

```python
def forward(self, xs, ts):
  for layer in self.layers:
    xs = layer.forward(xs)
  loss = self.loss_layer.forward(xs, ts)
  return loss

def backward(self, dout=1):
  dout = self.loss_layer.backward(dout)
  for layer in reversed(self.layers):
    dout = layer.backward(dout)
  return dout

def reset_state(self):
  self.rnn_layer.reset_state()
```

## 5.2 언어 모델의 평가

언어 모델은 주어진 과거 단어(정보)로부터 다음에 출현할 단어의 확률분포를 출력한다. 이때 언어 모델의 예측 성능을 평가하는 척도로 혼란도(perplexity)를 자주 이용한다.

혼란도(perplexity) : 간단히 말하면 '확률의 역수'이다.(데이터 수가 하나일 때에 정확히 일치한다.) 작을수록 좋은 값이다.

분기수(number of branches): 다음에 취할 수 있는 선택사항의 수(다음에 출현할 수 있는 단어의 후보 수)

- 예시

분기수가 1.25 -> 다음에 출현할 수 있는 단어의 후보를 1개 정도로 좁혔다(좋은 모델)

분기수가 5 -> 후보가 아직 5개(나쁜 모델)

입력 데이터가 여러 개일 때

![CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2025.png](CHAPTER%205%20%E1%84%89%E1%85%AE%E1%86%AB%E1%84%92%E1%85%AA%E1%86%AB%20%E1%84%89%E1%85%B5%E1%86%AB%E1%84%80%E1%85%A7%E1%86%BC%E1%84%86%E1%85%A1%E1%86%BC(RNN)%20ad30bc6e5a9e4ce2bcd7df0bc479a626/Untitled%2025.png)

N:데이터의 총개수

tn: 원핫 벡터로 나타낸 정답 레이블

tnk: n개째 데이터의 k번째 값

ynk: 확률분포(신경망에서sms Softmax의 출력)

L: 신경망의 손실. 교차 엔트로피 오차를 뜻하는 식과 같은 식

## 5.3 RNNLM의 학습 코드

```python
import sys
sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파라미터 설정
batch_size = 10
wordvec_size = 100
hidden_size = 100 # RNN의 은닉 상태 벡터의 원소 수
time_size = 5     # Truncated BPTT가 한 번에 펼치는 시간 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기(전체 중 1000개만)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # 입력
ts = corpus[1:]   # 출력(정답 레이블)
data_size = len(xs)
print('말뭉치 크기: %d, 어휘 수: %d' % (corpus_size, vocab_size))

# 학습 시 사용하는 변수
max_iters = data_size // (batch_size * time_size)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 모델 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 미니배치의 각 샘플의 읽기 시작 위치를 계산
jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]
#각 미니배치가 데이터를 읽기 시작하는 위치를 계산해 offsets에 저장한다.
#offsets의 각 원소에 데이터를 읽는 시작 위치가 담기게 된다.

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # 미니배치 취득
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
                #말뭉치를 읽는 위치가 말뭉치 크기를 넘어설 경우 말뭉치의 처음으로
                #돌아오기 위해서 말뭉치의 크기로 나눈 나머지를 인덱스로 사용한다.
            time_idx += 1
            #time_idx를 1씩 늘리면서 말뭉치에서 time_idx위치의 데이터를 얻는다.

        # 기울기를 구하여 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 에폭마다 퍼플렉서티 평가
    ppl = np.exp(total_loss / loss_count)
    print('| 에폭 %d | 퍼플렉서티 %.2f'
          % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 그래프 그리기
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
```

## 5.4 RNNLM의 Trainer 클래스

RNNLM 학습을 수행해주는 RNNlmTrainer 클래스

```python
import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파라미터 설정
batch_size = 10
wordvec_size = 100
hidden_size = 100  # RNN의 은닉 상태 벡터의 원소 수
time_size = 5  # RNN을 펼치는 크기
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000  # 테스트 데이터셋을 작게 설정
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
xs = corpus[:-1]  # 입력
ts = corpus[1:]  # 출력（정답 레이블）

# 모델 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

trainer.fit(xs, ts, max_epoch, batch_size, time_size) #학습 수행
# 1 미니배치를 '순차적'으로 만들어
# 2 모델의 순전파와 역전파를 호출하고
# 3 옵티마이저로 가중치를 갱신하고
# 4 퍼플렉시티를 구한다.
trainer.plot()
```

# 6. 정리

- RNN은 순환하는 경로가 있고 이를 통해 내부에 '은닉 상태'를 기억할 수 있다.
- RNN의 순환 경로를 펼침으로써 다수의 RNN 계층이 연결된 신경망으로 해석할 수 있으며, 보통의 오차역전파법으로 학습할 수 있다.(=BPTT)
- 긴 시계열 데이터를 학습할 때는 데이터를 적당한 길이씩 모으고(이를 '블록'이라 한다)블록 단위로 BPTT에 의한 학습을 수행한다.(=Truncated BPTT)
- Truncated BPTT에서는 역전파의 연결만 끊는다.
- Truncated BPTT에서는 순전파의 연결을 유지하기 위해 데이터를 '순차적'으로 입력해야 한다.
- 언어 모델은 단어 시퀀스를 확률로 해석한다.
- RNN 계층을 이용한 조건부 언어 모델은 (이론적으로는) 그때까지 등장한 모든 단어의 정보를 기억할 수 있다.