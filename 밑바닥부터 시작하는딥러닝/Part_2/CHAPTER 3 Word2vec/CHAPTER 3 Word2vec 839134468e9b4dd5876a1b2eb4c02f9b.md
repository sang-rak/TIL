# CHAPTER 3 Word2vec

- **목차**

# 1. 추론 기반 기법과 신경망

단어를 벡터로 표현하는 방법은 크게 두 부분이 있다.

1. 통계 기반 기법
2. 추론 기반 기법

단어의 의미를 얻는 방식은 서로 크게 다르지만, 그 배경에는 모두 분포 가설이 있다.

이번 절에서는 통계 기반 기법의 문제를 지적하고 그 대안인 추론 기반 기법의 이점을 거시적 관점에서 설명한다.

## 1.1 통계 기반 기법의 문제점

지금까지 본 것 처럼 통계 기반 기법에서는 주변 단어의 빈도를 기초로 단어를 표현했다. 구체적으로는 단어의 동시 발생 행렬을 만들고 그 행렬에 SVD를 적용하여 밀집벡터(단어의 분산 표현)를 얻었다.

그러나 이 방식은 대규모 말뭉치를 다룰 때 문제가 발생한다.

현업에서 다루는 말뭉치의 어휘 수는 어마어마하다.

이런 거대 행렬에 SVD를 적용하는 일은 현실적이지 않다.

- SVD를 n*n행렬에 적용하는 비용은 O(n3)이다.

통계 기반 기법은 말뭉치 전체의 통계(동시 발생 행렬과 PPMI 등) 를 이용해 단 1회의 처리(SVD 등)으로 단어의 분산 표현을 얻는다.

한편, 추론 기반기법에서는 예컨대 신경망을 이용하는 경우는 미니 배치로 학습하는 것이 일반적이다.

미니배치 학습에서는 신경망이 한번에 소량(미니배치의 학습 샘플)씩 반복해서 학습하며 가중치를 갱신한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled.png)

통계 기반 기법은 학습 데이터를 한꺼번에 처리한다. (배치학습)

추론 기반 기법은 학습 데이터를 일부를 사용하여 순차적으로 학습한다. (미니배치 학습)

말뭉치의 어휘 수가 많아 SVD등 계산량이 큰 작업을 처리하기 어려운 경우에도 신경망을 학습시킬 수 있다는 의미이다. 데이터를 작게 나눠 학습하기 때문이다.

## 1.2 추론 기반 기법 개요

추론 기반 기법에서는 추론이 주된 작업이다.

추론이란, 주변 단어(맥락)가 주어졌을 때 ? 에 무슨 단어가 들어가는지를 추측하는 작업이다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%201.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%201.png)

추론 문제를 풀고 학습하는 것이 추론 기반 기법이 다루는 문제이다.

이러한 추론 문제를 반복해서 풀면서 단어의 출현 패턴을 학습하는 것이다.

모델 관점에서 보면 추론문제는 다음과 같다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%202.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%202.png)

추론 기반 기벙에는 어떠한 모델이 등장한다.

우리는 이 모델로 신경망을 사용한다.

모델은 맥락 정보를 입력받아 출현 할 수 있는 각 단어의 출현 확률을 출력한다.

이러한 틀 안에서 말뭉치를 사용해 모델이 올바른 추측을 내놓도록 학습시킨다.

그리고 그 학습의 결과로 단어의 분산 표현을 얻는 것이 추론 기반 기법의 전체 그림이다.

추론 기반 기법도 통계 기반 기법처럼 가설에 기초한다.

분포 가설이란 단어의 의미는 주변 단어에 의해 형성된다는 가설로 이를 추측문제로 귀결 시켰다.

이처럼 두 기법 모두 분포 가설에 근거하는 단어의 동시 발생 가능성을 얼마나 잘 모델링 하는가가 중요한 연구 주제이다.

## 1.3 신경망에서의 단어 처리

단어를 있는 그대로 처리할 수 없으니 고정길이의 벡터로 변환해야 한다.

이때 사용하는 대표적인 방법이 단어를 원핫 표현으로 변환하는 것이다.

원핫 표현이란 벡터의 원소 중 하나만 1이고, 나머지는 모두 0인 벡터를 말한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%203.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%203.png)

단어를 원핫 표현으로 변환하는 방법

- 먼저 총 어휘 수만큼의 원소를 갖는 벡터를 준비하고
- 인덱스가 단어 ID와 같은 원소를 1로 나머지는 모두 0으로 설정한다.

    이처럼 단어를 고정 길이 벡터로 변환하면, 신경망의 입력층은 뉴련의 수를 고정할 수 있다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%204.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%204.png)

단어를 벡터로 나타낼 수 있고, 신경망을 구성하는 계층들은 벡터를 처리할 수 있다.

다시 말해 단어를 신경망으로 처리 할 수 있다는 뜻이다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%205.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%205.png)

화살표에는 가중치(매개변수)가 존재하여 입력층 뉴런과의 가중합이 은닉층 뉴런이 된다.

참고로 이번 장에서 사용하는 완전연결 계층에서는 편향을 생략 했다.

편향을 이용하지 않은 완전 연결 계층은 행렬 곱 계산에 해당한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%206.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%206.png)

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%207.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%207.png)

# 2. 단순한 Word2vec

이번 절에서 사용할 신경망은 word2vec에서 제안하는 CBOW, continuous bag-of-words 모델이다.

## 2.1 CBOW 모델의 추론 처리

CBOW 모델은 맥락으로 부터 타깃을 추측하는 용도의 신경망이다.

타깃은 중앙 단어이고, 그 주변 단어들이 맥락이다.

우리는 이 CBOW모델이 가능한 정확하게 추론하도록 훈련시켜서 단어의 분산 표현을 얻어낼 것이다.

CBOW 모델의 입력은 맥락이다.

가장 먼저 이 맥락을 원핫 표현으로 변환하여 CBOW모델이 처리할 수 있도록 준비한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%208.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%208.png)

그림에서 입력층이 2개인 이유는 맥락으로 고려할 단어를 2개로 정했기 때문이다.

은닉층의 뉴런은 입력층의 완전연결 계층에 의해 변환 된 값이 되는데 입력층이 여러 개이면 전체를 평균하면 된다.

그림에서 출력층의 뉴런은 총 7개인데 중요한 것은 이 뉴런 하나하나가 각각의 단어에 대응한다는 점이다.

그리고 출력층 뉴런은 각 단어의 점수를 뜻하며 값이 높을수록 대응 단어의 출현확률도 높아진다. 여기서 점수란 확률로 해석되기 전의 값이고 이 점수에 소프트맥스 함수를 적용해서 확률을 얻을 수 있다.

점수를 Softmax 계층에 통과시킨 후의 뉴런을 출력층이라고도 한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%209.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%209.png)

은닉층의 뉴런 수를 입력층의 뉴런 수 보다 적게 하는 것이 중요한 핵심이다. 이렇게 해야 은닉층에는 단어 예측에 필요한 정보를 간결하게 담게 되며 결과적으로 밀집벡터 표현을 얻을 수 있다.

이때 은닉층 정보는 인간이 이해할 수 없는 코드로 쓰여있다. (인코딩)

한편, 은닉층의 정보로부터 원하는 결과를 얻는 작업은 디코딩이라고 한다.

즉, 디코딩이란 인코딩된 정보를 인간이 이해할 수 있는 표현으로 복원하는 작업이다.

지금까지 우리는 CBOW 모델을 뉴런 관점에서 그렸다.

이번에는 계층 관점에서 그려보자

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2010.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2010.png)

편향을 사용하지 않은 완전 연결 계층의 처리는 MatMul 계층의 순전파와 같다.

MatMul 계층은 내부에서 행렬 곱을 계산한다.

CBOW 모델의 추론 처리를 구현해보자.

추론 처리란 점수를 구하는 처리를 말한다.

```python
# ch03/cbow_predict.py
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul

# 샘플 맥락 데이터
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 계층 생성
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)
print(s)
```

이상이 CBOW 모델의 추론 과정이다.

CBOW 모델은 활성화함수를 사용하지 않는 간단한 구성의 신경망이다.

## 2.2 CBOW 모델의 학습

지금까지 설명한 CBOW 모델은 출력층에서 각 단어의 점수를 출력했다.

이 점수에 소프트맥스 함수를 적용하면 확률을 얻을 수 있다.

이 확률은 맥락(전후 단어)이 주어졌을 때 그 중앙에 어떤 단어가 출현하는지를 나타낸다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2011.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2011.png)

CBOW 모델의 학습에서는 올바른 예측을 할 수 있도록 가중치를 조정하는 일을 한다.

그 결과, 입력층/출력층 가중치에 단어의 출현 패턴을 파악한 벡터가 학습된다.

이렇게 얻은 CBOW 모델과 skip-gram 모델로 얻을 수 있는 단어의 분산 표현은 단어의 의미 면에서나 문법 면에서 모두 우리의 직관에 부합한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2012.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2012.png)

추론 처리를 수행하는 CBOW 모델에 Softmax 계층과 Cross Entropy 계층을 추가한 것 만으로도 손실을 얻을 수 있다.

이상이 CBOW 모델의 손실을 구하는 계산 흐름이자, 이 신경망의 순방향 전파이다.

Softmax계층과 Cross Entropy Error 계층은 Softmax with Loss 계층 하나로 구현 할 수 있다.

따라서 앞으로 우리가 구현할 신경망의 정확한 모습은 아래와 같다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2013.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2013.png)

## 2.3 word2vec의 가중치와 분산 표현

word2vec에서 사용되는 신경망에는 두 가지 가중치가 있다.

입력 층 완전연결 계층의 가중치와 출력 층 완전연결계층의 가중치이다.

그리고 입력층 가중치의 각 행이 각 단어의 분산 표현에 해당한다.

다만, 출력층 가중치는 각 단어의 분산 표현이 열 방향(수직방향)으로 저장된다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2014.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2014.png)

그러면 최종적으로 이용하는 단어의 분산 표현으로는 어느 쪽 가중치를 사용하면 좋을까 ?

선택지는 3가지

1. 입력 측의 가중치만 이용
2. 출력 측의 가중치만 이용
3. 양쪽 가중치를 모두 이용

word2vec 특히 skip-gram 모델에서는 입력측 가중치만 이용하는 것이 가장 대중적이다.

# 3. 학습 데이터 준비

## 3.1 맥락과 타깃

word2vec에서 이용하는 신경망의 입력은 맥락이다.

그리고 정답 레이블은 맥락에 둘러싸인 중앙의 단어, 즉 타깃이다.

우리가 해야 할 일은 신경망에 맥락을 입력했을 때 타깃이 출현할 확률을 높이는 것이다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2015.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2015.png)

말뭉치로부터 맥락과 타깃을 만드는 함수를 구현하기 전에,

앞 장의 내용을 복습해보자

우선 말뭉치 텍스트를 단어 ID로 변환해야한다.

```python

import sys
sys.path.append('..')
from common.util import preprocess

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
print(corpus)

print(id_to_word)
```

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2016.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2016.png)

맥락과 타깃을 만드는 함수를 구현해보자.

```python
# common/util.py
def create_co_matrix(corpus, vocab_size, window_size=1):
    '''동시발생 행렬 생성
    :param corpus: 말뭉치(단어 ID 목록)
    :param vocab_size: 어휘 수
    :param window_size: 윈도우 크기(윈도우 크기가 1이면 타깃 단어 좌우 한 단어씩이 맥락에 포함)
    :return: 동시발생 행렬
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix
```

이것으로 말뭉치로부터 맥락과 타깃을 만들어냈다.

나중에 이를 CBOW 모델에 넘겨주면 된다.

맥락과 타깃의 원소는 여전히 단어 ID이다.

이어서 이를 원핫 표현으로 변환해보자.

## 3.2 원핫 표현으로 변환

맥락과 타깃을 원핫 표현으로 바꿔보자.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2017.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2017.png)

맥락과 타깃을 단어 ID에서 원핫 표현으로 변환하면 된다.

```python
# common/utill.py
def convert_one_hot(corpus, vocab_size):
    '''원핫 표현으로 변환
    :param corpus: 단어 ID 목록(1차원 또는 2차원 넘파이 배열)
    :param vocab_size: 어휘 수
    :return: 원핫 표현(2차원 또는 3차원 넘파이 배열)
    '''
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot
```

```python
# 데이터 준비 과정
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.util import preprocess, create_contexts_target, convert_one_hot

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

contexts, target = create_contexts_target(corpus, window_size=1)

vocab_size = len(word_to_id)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)
```

# 4. CBOW 모델 구현

CBOW 모델을 구현해보자.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2018.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2018.png)

```python
# ch03/simple_cbow.py
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in
```

신경망의 순전파인 forward() 메서드를 구현한다.

```python
def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
```

마지막으로 역전파인 backward()를 구현한다.

```python
def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None
```

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2019.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2019.png)

## 4.1 학습 코드 구현

CBOW 모델의 학습은 일반적인 신경망의 학습과 완전히 같다. 학습 데이터를 준비해 신경망을 입력한 다음 기울기를 구하고 가중치 매개변수를 순서대로 갱신해간다.

```python
# ch03/train.py
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)
target = convert_one_hot(target, vocab_size)
contexts = convert_one_hot(contexts, vocab_size)

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()
```

매개변수 갱신 방법:SGD, AdaGrad 등 중 Adam을 선택한다.

```python
# common/optimizer.py

# coding: utf-8
import sys
sys.path.append('..')
from common.np import *

class SGD:
    '''
    확률적 경사하강법(Stochastic Gradient Descent)
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]

class Momentum:
    '''
    모멘텀 SGG(Momentum SGD)
    '''
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] = self.momentum * self.v[i] - self.lr * grads[i]
            params[i] += self.v[i]

class Nesterov:
    '''
    네스테로프 가속 경사(NAG; Nesterov's Accelerated Gradient) (http://arxiv.org/abs/1212.0901)
    '네스테로프 모멘텀 최적화'라고도 한다.
    '''
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = []
            for param in params:
                self.v.append(np.zeros_like(param))

        for i in range(len(params)):
            self.v[i] *= self.momentum
            self.v[i] -= self.lr * grads[i]
            params[i] += self.momentum * self.momentum * self.v[i]
            params[i] -= (1 + self.momentum) * self.lr * grads[i]

class AdaGrad:
    '''
    AdaGrad
    '''
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] += grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)

class RMSprop:
    '''
    RMSprop
    '''
    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = []
            for param in params:
                self.h.append(np.zeros_like(param))

        for i in range(len(params)):
            self.h[i] *= self.decay_rate
            self.h[i] += (1 - self.decay_rate) * grads[i] * grads[i]
            params[i] -= self.lr * grads[i] / (np.sqrt(self.h[i]) + 1e-7)

class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
```

Train 클래스는 신경망을 학습시킨다.

학습데이터로부터 미니 배치를 선택한 다음, 신경망을 입력해 기울기를 구하고 그 기울기를 Optimizer에 넘겨 매개변수를 갱신하는 일련의 작업을 수행한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2020.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2020.png)

학습이 끝난 후의 가중치 매개변수를 살펴보자.

입력 측 MatMul 계층의 가중치를 꺼내 실제 내용을 확인해보자.

```python
word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

# you [-0.9031807 -1.0374491 -1.4682057 -1.321632 0.931256]
# say [-1.2172916 -1.2620505 -0.07845993 0.07709391 -1.2389531]
```

결과: 단어를 밀집 벡터로 나타낼 수 있다.

이 밀집 벡터가 단어의 분산 표현이다.

학습이 잘 이루어졌으니 이 분산 표현은 단어의 의미를 잘 파악한 벡터 표현이라 기대할 수 있다.

현 시점의 CBOW 모델은 처리 효율 면에서 몇 가지 문제가 있다.

다음 4장에서는 현재의 단순한 CBOW 모델을 개선하여 진짜 CBOW 모델을 구현할 계획이다.

# 5. word2vec 보충

지금까지 word2vec의 CBOW 모델을 자세히 살펴봤다.

이번 절에서는 지금까지 말하지 못한 word2vec에 관한 중요한 주제 몇 개를 보충해보자.

우선은 CBOW 모델을 확률 관점에서 다시 살펴보자.

## 5.1 CBOW 모델과 확률

확률의 표기법을 간단하게 살펴보자

확률 P()

동시 확률 P(A, B) A와 B가 동시에 일어날 확률.

사후 확률 P(A|B) 사건이 일어난 후의 확률

B라는 정보가 주어졌을 때 A가 일어날 확률.

그럼 CBOW 모델을 확률 표기법으로 기술해보자.

CBOW 모델이 하는일은 맥락을 주면 타깃 단어가 출현할 확률을 출력하는 것이다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2021.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2021.png)

CBOW 모델은 다음 식을 모델링하고 있다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2022.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2022.png)

위 식을 이용하면 CBOW 모델의 손실 함수도 간결하게 표현할 수 있다.

교차 엔트로피 오차를 적용해보자.

다음 식을 유도할 수 있다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2023.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2023.png)

이 식을 보듯, CBOW 모델의 손실 함수는 단순히 확률식에 log를 취한다음 마이너스를 붙이면 된다. (음의 로그 가능) 덧붙여 위의 식은 샘플 데이터 하나에 대한 손실 함수이며, 이를 말뭉치 전체로 확장하면 다음 식이 된다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2024.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2024.png)

CBOW 모델의 학습이 수행하는 일은 손실 함수 식을 가능한 작게 만드는 것이다.

그리고 이때의 가중치 매개변수가 우리가 얻고자 하는 단어의 분산 표현이다.

## 5.2 skip-gram 모델

word2vec은 2개의 모델을 제안하고 있다.

1. CBOW 모델
2. skip-gram 모델

skip-gram은 CBOW에서 다루는 맥락과 타깃을 역전시킨 모델

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2025.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2025.png)

CBOW모델은 맥락이 여러 개 있고, 여러 맥락으로부터 중앙의 단어(타깃)를 추측한다.

한편, skip-gram 모델은 중앙의 단어(타깃)로부터 주변의 여러 단어(맥락)를 추측한다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2026.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2026.png)

그림을 보듯 skip-gram 모델의 입력층은 하나이다.

한편 출력층은 맥락의 수만큼 존재한다.

따라서 각 출력층에서는 Softmax with Loss 계층 등을 이용해 개별적으로 손실을 구하고 이 개별 손실들을 모두 더한 값을 최종 손실로 한다.

skip-gram 모델을 확률 표기로 나타내면 다음 식과 같다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2027.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2027.png)

skip-gram 모델에선느 맥락의 단어들 사이에 관련성이 없다고 가정하고 다음과 같이 분해한다.(조건부 독립)

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2028.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2028.png)

위 식을 교차 엔트로피 오차에 적용하여 skip-gram 모델의 손실 함수를 유도할 수 있다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2029.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2029.png)

위 식에서 알 수 있듯, skip-gram 모델의 손실 함수는 맥락별 손실을 구한 다음 모두 더한다.

위 식은 샘플 데이터 하나짜리 skip-gram 의 손실 함수이다.

이를 말뭉치 전체로 확장하면 skip-gram 모델의 손실 함수는 다음과 같다.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2030.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2030.png)

위식을 CBOW 모델의 식과 비교해보자.

![CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2031.png](CHAPTER%203%20Word2vec%20839134468e9b4dd5876a1b2eb4c02f9b/Untitled%2031.png)

skip-gram 모델은 맥락의 수만큼 추측하기 때문에 그 손실 함수는 각 맥락에서 구한 손실의 총합이어야 한다.

반면, CBOW 모델은 타깃 하나의 손실을 구한다.

그렇다면 CBOW 모델과 skip-gram 모델중 어느 것을 사용해야 할까?

답은 skip-gram

단어 분산 표현의 정밀도 면에서 skip-gram 모델의 결과가 더 좋은 경우가 많기 때문이다.

특히 말뭉치가 커질수록 저빈도 단어와 유추 문제의 성능 면에서 skip-gram 모델이 더 뛰어난 경향이 있다.

반면 학습 속도 면에서는 CBOW 모델이 더 빠르다.

skip-gram 모델은 손실을 맥락의 수만큼 구해야 해서 계산 비용이 그만큼 커지기 때문이다.

다행히 CBOW 모델의 구현을 이해할 수 있다면 skip-gram 모델의 구현도 특별히 어려울게 없다.

```python
# ch03/simple_skip_gram.py
import sys
sys.path.append('..')
import numpy as np
from common.layers import MatMul, SoftmaxWithLoss

class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2
        return loss

    def backward(self, dout=1):
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
```

## 5.3 통계 기반 vs 추론 기반

통계 기반 기법에서는 주로 단어의 유사성이 인코딩된다.

한편 word2vec 특히 skip-gram 모델에서는 단어의 유사성은 물론 한층 복잡한 단어 사이의 패턴까지도 파악되어 인코딩된다.

추론 기반 기법이 통계 기반 기법보다 정확하다고 흔히 오해하곤 한다.

하지만 단어의 유사성을 정량 평가해본 결과 추론 기반과 통계 기반 기법의 우열을 가릴 수 없었다고 한다.

추론 기반 기법과 통계 기반 기법은 서로 관련되어있다.

word2vec이후 추론 기반 기법과 통계 기반 기법을 융합한 GloVe 기법이 등장했다.

GloVe의 기본 아이디어는 말뭉치 전체의 통계 정보를 손실함수에 도입해 미니배치 학습을 하는것이다.

# 6. 정리

- 추론기반기법은 추측하는 것이 목적이며, 그 부산물로 단어의 분산 표현을 얻을 수 있다.
- word2vec은 추론 기반 기법이며 단순한 2층 신경망이다.
- word2vec은 skip-gram모델과 CBOW 모델을 제공한다.
- CBOW 모델은 여러 단어(맥락)로부터 하나의 단어(타깃)를 추측한다.
- 반대로 skip-gram 모델은 하나의 단어(타깃)로부터 다수의 단어(맥락)를 추측한다.
- word2vec은 가중치를 다시 학습할 수 있으므로, 단어의 분산 표현 갱신이나 새로운 단어 추가를 효율적으로 수행할 수 있다.