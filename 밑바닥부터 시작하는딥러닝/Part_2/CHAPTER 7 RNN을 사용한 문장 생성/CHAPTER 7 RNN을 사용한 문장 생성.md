# CHAPTER 7 RNN을 사용한 문장 생성

# 1. 언어 모델을 사용한 문장 생성

- 언어 모델은 다양한 애플리케이션에 활용 할 수 있다.
- 예시
    - 기계번역
    - 음성인식
    - 문장 생성 등

## 1.1 RNN을 사용한 문장 생성의 순서

- LSTM 계층을 이용해 언어모델을 구현 - 신경망 구성
- 시계열 데이터를 (T개분 만큼) 모아 처리하는 Time LSTM 과 Time Affine 계층 만듬

언어 모델: 오른쪽은 시계열 데이터를 한꺼번에 처리하는 Time 계층을 사용 왼쪽은 같은 구성을 펼친 모습

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled.png)

언어 모델은 다음에 출현할 단어의 확률 분포를 출력한다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%201.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 1.png)

- 언어 모델은 지금까지 주어진 단어들에서 다음에 출현하는 단어의 확률 분포를 출력한다.
- 다음 단어를 새로 생성하려면
    1. 확률이 가장높은 단어를 선택하는 방법
        - 확률이 가장 높은 단어를 선택 결과가 일정하게 정해진다.
    2. '확률적'으로 선택하는 방법도 생각 할 수 있음
        - 확률이 높은 단어는 선택되기 쉽고 확률이 낮은 단어는 선택되기 어려움
        - 선택되는 단어(샘플링 단어)가 매번 다를 수 있다.

## 1.2 문장 생성 구현

```python
import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm

class RnnlmGen(Rnnlm):
    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x)
            p = softmax(score.flatten())

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids
```

- generate 메서드는 가장먼저 model.predict(x)를 호출해 각 단어의 점수 출력 (정규화 전의 점수)
- p = softmax(score) 코드에서는 점수들을 소프트맥스 함수를 이용해 정규화한다.
- 목표로 하는 확률 분포 p 얻음 - 그 후 확률 분포 p로 단어 샘플링

RnnlmGen 클래스 사용해 문장 생성

```python
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
model.load_params('../ch06/Rnnlm.pkl')

# start 문자와 skip 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)
```

## 1.3 더 좋은 문장으로

좋은 언어 모델이 있으면 좋은 문장을 기대할 수 있다.

앞장에서 더 좋은 언어 모델을 BetterRnnln 라는 클래스로 구현했다.

여기에 문장생성 기능을 추가하겠다.

이모델을 한 단계 더 개선하고 한층 더 큰 말뭉치를 사용하면 더 자연스러운 문장을 생성해줄 것이다.

# 2. seq2seq

## 2.1 seq2seq의 원리

- 2개의 RNN을 이용
- Encoder-Decoder 모델
    - 2개의 모듈, Encoder와 Decoder가 등장한다.
    - Encoder는 입력 데이터를 인코딩(부호화)한다.
    - Decoder는 인코딩된 데이터를 디코딩(복호화)한다.

Encoder와 Decoder가 번역을 수행하는 예

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%202.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 2.png)

- Encoder와 Decoder가 협력하여 시계열 데이터를 다른 시계열 데이터로 변환한다.
- Encoder와 Decoder로는 RNN를 사용할 수 있다.

Encoder를 구성하는 계층

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%203.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 3.png)

- Encoder는 RNN을 이용해 시계열 데이터를 h라는 은닉 상태 벡터로 변환한다.
- RNN으로써의 LSTM을 이용했지만, '단순한 RNN'이나 GPU등도 물론 이용 가능하다.
- Encoder가 출력하는 벡터 h는 LSTM 계층의 마지막 은닉상태다.
- 은닉상태 h에 입력 문장(출발어)을 번역하는데 필요한 정보가 인코딩 된다.
- LSTM의 은닉 상태 벡터 h는 고정 길이 벡터다
- 인코딩 한다는것은 임의 길이의 문장을 고정 길이 벡터로 변환하는 작업이다.

Encoder는 문장을 고정 길이 벡터로 인코딩한다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%204.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 4.png)

Decoder를 구성하는 계층

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%205.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 5.png)

- LSTM 계층이 벡터 h를 입력 받는 점이 다르고 나머지는 같다.
- <eos>는 구분 기호(특수 문자)다. '구분자'이며 문장 생성의 시작을 알리는 신호로 이용된다.

seq2seq 전체 계층 구성

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%206.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 6.png)

- LSTM의 은닉 상태가 Encoder 와 Decoder를 이어주는 '가교'가 된다.
- 순전파 떄는 Encoder에서 인코딩 된 정보가 LSTM 계층의 은닉 상태를 통해 Decoder에 전해진다.

## 2.2 시계열 데이터 변환용 장난감 문제

- 57+5 와 같은 문자열을 seq2seq에 건내면 62라는 정답을 내놓도록 학습시킨다.
- 머신러닝을 평가하고자 만든 간단한 문제를 ' 장난감 문제(tory problem)이라고 한다.

seq2seq에 덧셈 예제들을 학습시킨다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%207.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 7.png)

- seq2seq는 덧셈 논리에 대해 아무것도 모르니 학습을 시켜야 한다.
- word2vec이나 언어 모델에서는 문장을 '단어' 단위로 분할해왔다.
    - 하지만 모든 문장에서 단어 단위로 분할 해야 하는 것은 아니다.
    - 이번에는 '문자' 단위로 분할한다.
    - 예) ['5', '7', '+', '5']라는 리스크로 처리

## 2.3 가변 길이 시계열 데이터

- '미니배치 처리'를 하려면 무언가 추가해야 한다.
- '미니배치 처리'로 학습할 때는 다수의 샘플을 한꺼번에 처리하며 한 미니배치에 속한 샘플들의 데이터 형상이 모두 똑같아야 한다.
- 시계열 데이터를 미니배치로 학습하기 위한 가장 단순한 방법은 패딩 이다.

미니배치학습을 위해 '공백문자로 패딩을 수행하여 입력,출력 데이터의 크기를 통일한다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%208.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 8.png)

- 임의로 0~999 사이의 숫자 2개만 더하는 것으로 설정
- '+'까지 포함하면 최대 문자 수는 7이 된다.
- 출력 데이터는 최대 4문자이고 출력 데이터는 구분하기 위해 앞에(_)를 붙인다.
- 패딩을 적용해 데이터를 통일 시키면 가변 길이의 시계열 데이터도 처리할 수 있다.
- 원래는 존재하지 않는 패딩용 문자(공백)까지 처리해야 하므로 정확성이 중요하다면 seq2seq에 패딩전용 처리를 추가해야 한다.

## 2.4 덧셈 데이터셋

지금부터 사용할 덧셈 학습 데이터는 예제 총 5000개가 들어 있고 다음 그림과 같은 형태이다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%209.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 9.png)

덧셈 데이터셋 사용하는예

```python
import sys
sys.path.append('..')
from dataset import sequence

(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('addition.txt', seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)
# (45000, 7) (45000, 5)
# (5000, 7) (5000, 5)

print(x_train[0])
print(t_train[0])
# [ 3  0  2  0  0 11  5]
# [ 6  0 11  7  5]

print(''.join([id_to_char[c] for c in x_train[0]]))
print(''.join([id_to_char[c] for c in t_train[0]]))
# 71+118
# _189
```

# 3. seq2seq 구현

## 3.1 Encoder 클래스

- Encoder 크랠스는 아래그림처럼 문자열을 입력 받아 벡터 h로 변환한다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2010.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 10.png)

- RNN을 이용해 Encoder를 구성 - LSTM 계층이용

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2011.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 11.png)

- Encoder 클래스는 Embedding 계층과 LSTM계층으로 구성
- Embedding 계층에서는 문자(문자 ID)를 문자 벡터로 변환한다.
- 문자벡터가 LSTM계층으로 입력된다.
- Encoder에서는 LSTM의 은닉 상태만을 Decoder에 전달한다
    - LSTM의 셀도 Decoder에 전달할 수 있지만, LSTM의 셀을 다른 계층에 전달하는 일은 일반적으로 흔하지 않다.
    - LSTM의 셀은 자신만 사용한다는 전제로 설계되었기 때문이다.

## 3.2 Decoder 클래스

- Encoder 클래스가 출력한 h를 받아 목적으로하는 다른 문자열을 출력

Encoder와 Decoder

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2012.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 12.png)

Decoder의 계층 구성 (학습시)

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2013.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 13.png)

- 정답데이터는 "_62", 입력데이터 = ['_','6','2','']로 주고 이에 대응하는 출력은 ['6','2','','']이 되도록 학습시킨다.

Decoder의 문자열 생성 순서: argmax노드는 Afiine 계층의 출력중 값이 가장 큰 원소의 인덱스(문자ID)를 반환 한다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2014.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 14.png)

- argmax노드는 최대값을 가진 원소의 인덱스(문자ID)를 선태갛는 노드
- Softmax 계층은 입력된 벡터를 정규화 한다.
    - 정규화 과정에서 벡터의 각 원소의 값이 달라진다
    - 대소 관계는 바뀌지 않는다.
    - 따라서 위의 그림은 Softmax계층을 생략할 수 있다.

    ![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2015.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 15.png)

Decoder 클래스는 Time Embedding, Time LSTM, Time Affine 3가지 계층으로 구성

```python
class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh
```

- Decoder클래스는 학습시와 문장 생성시의 동작이 다르다
- 앞의 forward()메서드는 학습할때 사용된다고 가정
- 문장 생성을 담당하는 generate() 메서드 구현

Decoder 클래스에 문장 생성을 담당하는 generte() 메서드를 살펴보자

```python

def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled
```

- generate() 메서드는 인수 3개를 받는다 - Encoder로 부터 받는 은닉상태 h 최초로 주어지는 문자 ID인 start_id 생성하는 문자수인 sample_size
- 여기서 문자를 1개씩 주고 Affine 계층이 출력하는 점수가 가장 큰 문자 ID를 선택하는 작업을 반복한다.

이번 문제에서는 Encoder 의 출력 h를 Decoder 의 Time LSTM 계층의 상태로 설정했다.

즉, Time LSTM 계층은 상태를 갖도록 한 것이다.

단, 한 번 설정된 이 은닉 상태느 재설정되지 않고, 즉 Encoder 의 h를 유지하면서 순전파가 이뤄진다.

## 3.3 Seq2seq 클래스

- Encoder 클래스와 Decoder 클래스를 연결하고 Time Sofitmax with Loss 계층을 이용해 손실을 계산

```python
class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled
```

## 3.4 seq2seq 평가

seq2seq 의 학습은 기본적인 신경망의 학습과 같은 흐름으로 이뤄진다.

1. 학습 데이터에서 미니배치를 선택하고

2. 매니배치로부터 기울기를 선택하고

3. 기울기를 사용하여 매개변수를 갱신한다.

Tainer 클래스를 사용해 이 규칙대로 작업을 수행한다.매 애폭마다 seq2seq 가 테스트 데이터를 풀게 하여(문자열을 생성하여) 학습 중간중간 정답률을 측정한다.

다음은 seq2seq 학습 코드이다.

```python
import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
from dataset import sequence
from common.optimizer import Adam
from common.trainer import Trainer
from common.util import eval_seq2seq
from seq2seq import Seq2seq
from peeky_seq2seq import PeekySeq2seq

# 데이터셋 읽기
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# 하이퍼파라미터 설정
vocab_size = len(char_to_id)
wordvec_size = 16
hideen_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

model = Seq2seq(vocab_size, wordvec_size, hideen_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
    trainer.fit(x_train, t_train, max_epoch=1,
                batch_size=batch_size, max_grad=max_grad)

    correct_num = 0
    for i in range(len(x_test)):
        question, correct = x_test[[i]], t_test[[i]]
        verbose = i < 10
        correct_num += eval_seq2seq(model, question, correct,
                                    id_to_char, verbose, is_reverse)

    acc = float(correct_num) / len(x_test)
    acc_list.append(acc)
    print('검증 정확도 %.3f%%' % (acc * 100))
```

기본적인 신경망의 학습용 코드와 같지만,평가 척도로 정답률을 사용했다.정확하게는 에폭마다 테스트 데이터의 문제 중 몇 개를 풀게 하여 올바르게 답했는지를 채점했다.

코드를 실행

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2016.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 16.png)

학습이 진행됨에 따라 출력되는 결과 몇 개를 선택해 보여준 것이다.

seq2seq 는 초기에는 정답을 잘 맞히지 못했지만, 학습을 거듭할수록 조금식 정답에 가까워지면서 몇 개씩은 맞히기 시작한다.

에폭마다의 정답률을 그래프로 살펴보자

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2017.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 17.png)

# 4. seq2seq 개선

이번에는 앞의 seq2seq를 세분화하여 학습 속도를 개선하고자 한다.

효과적인 기법이 몇 가지 있는데, 그중 두 가지 개선안을 확인해보자.

## 4.1 입력 데이터 반전

첫 번째 개선안은 아주 쉬운 트릭으로, 다음 그림과 같이 입력 데이터의 순서를 반전시킨다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2018.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 18.png)

입력 데이터를 반전시키는 트릭을 사용하면 많은 경우 학습 진행이 빨라져서, 결과적으로 최종 정확도가 좋아진다고 한다.

그럼 실제로 코드를 살펴보자.

```python
x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]
```

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2019.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 19.png)

물론 데이터를 반전시키는 효과는 어떤 문제를 다루느냐에 따라 다르지만, 대부분의 경우 더 좋은 결과로 이어진다.

그러면 왜 입력 데이터를 반전시키는 것만으로 학습의 진행이 빨라지고 정확도가 향상되는 걸까?이론적인 것은 잘 모르겠지만, 직관적으로는 **기울기 전파가 원활해지기 때문**이라고 생각한다.

예를 들어 "나는 고양이로소이다"를 "I am a cat"으로 번역하는 문제에서,'나'라는 단어가 'I'로 변환되는 과정을 생각해보자.이때 '나'로부터 'I'까지 가려면 '는','고양이','로소','이다'Rkwl chd 4 단어 분량의 LSTM 계층을 거쳐야 한다.따라서 역전파 시, 'I'로부터 전해지는 기울기가 '나'에 도달하기까지, 그 먼 거리만큼 영향을 더 받게 된다.

여기서 입력문을 반전시키면, 즉 "이다 로소 고양이 는" 순으로 바꾸면 어떻게 될까요?이제 '나'와 'I'는 바로 옆이 되었으니 기울기가 직접 전해진다.이처럼 입력 문장의 첫 부분에서는 반전 덕분에 대응하는 변환 후 단어와 가까우므로 (그런 경우가 많아지므로),기울기가 더 잘 전해져서 학습 효율이 좋아진다고 생각할 수 있다.다만, 입력 데이터를 반전해도 단어 사이의 평균적인 거리는 그대로이다.

## 4.2 엿보기

이어서 seq2seq 의 두 번째 개선이다.

주제로 곧장 들어가기 전에 seq2seq의 Encoder 동작을 한번 더 살펴보자.

Encoder는 입력 문장(문제 문장)을 고정 길이 벡터 h로 변환한다.이때 h 안에는 Decoder 에게 필요한 정보가 모두 담겨 있다.즉, h가 Decoder 에 있어서는 유일한 정보인 셈이다.

그러나 현재의 seq2eq 는 다음 그림과 같이 최초 시각의 LSTM 계층만이 벡터 h를 이용하고 있다.

이 중요한 정보인 h를 더 활용할 수는 없을까?

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2020.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 20.png)

여기서 seq2seq 의 두 번째 개선안이 등장한다.

중요한 정보가 담긴 Encoder 의 출력 h를 Decoder 의 다른 계층에게도 전해주는 것이다.

Decoder는 다음 그림과 같은 구성이 된다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2021.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 21.png)

그림과 같이 모든 시각의 Affine 계층과 LSTM 계층에 Encoder 의 출력 h를 전해준다.

앞 그림과 비교해보면, 기존에는 하나의 LSTM 만이 소유하던 중요 정보 h를 여러 계층(예에서는 8계층)이 공유함을 알 수 있다.

이는 집단 지성에 비유할 수 있다.

즉, 중요한 정보를 한 사람이 독점하는 것이 아니라, 많은 사람과 공유한다면 더 올바른 결정을 내릴 가능성이 커질 것이다.

이 개선안은 인코딩된 정보를 Decoder 의 다른 계층에도 전해주는 기법이다.달리 보면, 다른 계층도 인코딩된 정보를 엿본다고 해석할 수 있다.

엿보다를 영어로 peek 이라고 하기 때문에 이 개선을 더한 Decoder 를 Peeky Decoder 라고 한다.마찬가지로 Peeky Decoder 를 이용하는 seq2seq를 Peeky seq2seq 라고 한다.

위 그림에서는 LSTM 계층과 Affine 계층에 입력되는 벡터가 2개씩이 되었다.

이는 실제로는 두 벡터가 연결된 것을 의미한다.따라서 두 벡터를 연결시키는 concat 노드를 이용해 다음 그림처럼 그려야 정확한 계산 그래프가 된다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2022.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 22.png)

Peeky Decoder 클래스의 구현

```python
class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape
        N, H = h.shape

        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N, T, H)
        out = np.concatenate((hs, out), axis=2)

        out = self.lstm.forward(out)
        out = np.concatenate((hs, out), axis=2)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled
```

```python
class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
```

```python
model = PeekySeq2seq(vocab_size, wordvec_size, hideen_size)
```

결과는 다음 그림과 같다

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2023.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 23.png)

여기서 수행한 개선은 작은 개선이라 할 수 있다.큰 개선은 다음 장에서 추가할 계획이다.바로 어텐션이라는 기술로,seq2seq 를 극적으로 진화시킬 수 있다.

이번 절의 실험은 주의해야 한다.Peeky 를 이용하게 되면, 신경망은 가중치 매개변수가 커져서 계산량도 늘어난다.따라서 이번 절의 실험 결과는 커진 매개변수만큼의 핸디캡을 감안해야 한다.

또한 seq2seq 의 정확도는 하이퍼 파라미터에 영향을 크게 받는다.예제에서의 결과는 믿음직했지만, 실제 문제에서는 그 효과가 달라질 것이다.

# 5. seq2seq를 이용하는 애플리케이션

seq2seq 는 한 시계열 데이터를 다른 시계열 데이터로 변환한다.이 시계열 데이터를 변환하는 프레임워크는 다양한 문제에 적용할 수 있다.다음은 구체적인 예이다.

- 기계 번역: 한 언어의 문장을 다른 언어의 문장으로 변환
- 자동 요약: 긴 문장을 짧게 요약된 문장으로 변환
- 질의응답: 질문을 응답으로 변환
- 메일 자동 응답: 받은 메일의 문장을 답변 글로 변환

seq2seq 는 2개가 짝을 이루는 시계열 데이터를 다루는 문제에 이용할 수 있다.자연어 외에도 음성이나 영상 등에도 이용할 수 있다.얼핏 보기에는 seq2seq 가 적용될 수 없을 것 같은 문제라도 입력/출력 데이터를 전처리하면 seq2seq 를 적용할 수 있는 경우도 있다.

이번 절에서는 seq2seq 를 사용한 애플리케이션을 몇 가지 소개한다.

## 5.1 챗봇

챗봇은 사람과 컴퓨터가 텍스트로 대화를 나누는 프로그램이다.

챗봇에도 seq2seq를 사용할 수 있다.

대화라는 것은 상대의 말과 자신의 말로 구성되기 때문에 상대의 말을 자신의 말로 변환하는 문제로 볼 수 있다.

즉, 대화의 텍스트 데이터가 준비되면 그것으로 seq2seq 를 학습시킬 수 있다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2024.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 24.png)

대화를 보면, 기계(챗봇)은 훌륭하게 문제를 해결하고 있다.

VPN 연결이 되지 않은 사람을, 그 문제를 해결할 수 있는 URL 링크로 안내한 것이다.

물론 이 챗봇은 IT 헬프데스크에 한정된 문제를 대상으로 하기 때문에 범용적으로 사용할 수는 없다.

하지만 대화 기반으로 정답이나 힌트를 얻는 방식은 실용성이 높고 다양하게 응용하여 효과를 볼 수 있다.

## 5.2 알고리즘 학습

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2025.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 25.png)

## 5.3 이미지 캡셔닝

지금까지는 seq2seq 가 텍스트를 다루는 예만을 보았다.하지만 seq2seq 는 텍스트 외에도, 이미지나 음성 등 다양한 데이터를 처리할 수 있다.

이번절에서는 이미지를 문장으로 변환하는 이미지 캡셔닝을 살펴본다.

이미지 캡셔닝은 이미지를 문장으로 변환한다.이 문제도 다음 그림과 같이 seq2seq 의 틀에서 해결할 수 있다.

![CHAPTER%207%20RNN%E1%84%8B%E1%85%B3%E1%86%AF%20%E1%84%89%E1%85%A1%E1%84%8B%E1%85%AD%E1%86%BC%E1%84%92%E1%85%A1%E1%86%AB%20%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A1%E1%86%BC%20%E1%84%89%E1%85%A2%E1%86%BC%E1%84%89%E1%85%A5%E1%86%BC%20c13b4f6c1ba6463a9cbd60bab445cbfd/Untitled%2026.png](CHAPTER 7 RNN을 사용한 문장 생성.assets/Untitled 26.png)

그림은 우리에게 친숙한 신경망 구성이다.

지금까지와 다른 점은 Encoder 가 LSTM 에서 합성곱 신경망(CNN)으로 바뀐게 전부다.

겨우 LSTM을 CNN으로 대체한 것 만으로 seq2seq 는 이미지도 처리할 수 있다.

이 예에서는 이미지의 인코딩을 CNN이 수행한다. 이때 CNN의 최종 출력은 특징 맵이다.

특징 맵은 3차원(높이,폭,패털)이므로, 이를 Decoder 의 LSTM이 처리할 수 있도록 손질해야 한다.

그래서 CNN의 특징 맵을 1차원으로 평탄화한 후 완전연결인 Affine 계층에서 변환한다.

그런 다음 변환된 데이터를 Decoder 에 전달하면, 문장 생성을 수행할 수 있다.

위 그림의 CNN에 CGG나 ResNet 등의 입증된 신경망을 사용하고, 가중치로는 다른 이미지 데이터넷으로 학습을 끝낸 것을 이용한다.

이렇게 하면 좋은 인코딩을 얻을 수 있고, 좋은 문장을 생성할 수 있다.

이제 seq2seq 가 이미지 캡셔닝을 수행한 예를 몇 가지 살펴보자.

여기에 사용된 신경망은 앞 그림을 기초로 한 것이다.

# 6. 정리

- RNN을 이용한 언어 모델은 새로운 문장을 생성할 수 있다.
- 문장을 생성할 때는 하나의 단어(혹은 문자)를 주고 모델의 출력(확률분포)에서 샘플링하는 과정을 반복한다
- RNN을 2개 조합함으로써 시계열 데이터를 다른 시계열 데이터로 변환할 수 있다.