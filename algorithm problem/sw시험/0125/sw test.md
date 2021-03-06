># Problem 01
>
>시험 점수를 관리하는 코드를 작성하려고 한다.
>
>미리  수집한 과목 점수는 json 형식으로 저장 되어있다.
>
>이를 불러온느 코드는 미리 작성해 두었다.
>
>시험 점수 데이터 problem01_data.json 의 데이터는 다음과 같이 구성되어 있다고 할 때
>
>문제를 해결하시오.
>
>index : 0 1 2 3 
>
>정보 : python 점수, html 점수, javascript 점수,  project 점수



## problem01_01.py

전체 점수 중 최고점을 반환하는 함수 max_score을 완성하시오.

```python
import json

def max_score(scores):
    # 기본값을 시험점수 최저 값인 0으로 한다.
    default = 0 

    # 점수를 하나씩 불러온다
    for i in scores:
        # 불러온 점수가 이전 점수보다 높으면 바꿔준다.
        if i > default:
            default = i 
    # 가장 높은 점수를 반환 한다.
    return default
    # 여기에 코드를 작성합니다.


# 아래의 코드는 수정하지 않습니다.
if __name__ == '__main__':
    scores_json = open('problem01_data.json', encoding='UTF8')
    scores = json.load(scores_json)
    print(max_score(scores)) 
    # => 90
```



## problem01_02.py

전체 점수 중 60점 이상인 과목의 개수를 계산하는 함수 over를 완성하시오.

```python
import json

# 전체점수중 60점 이상인 과목의 개수를 count
def over(scores):
    # 과목의 갯수를 카운트하기위해 기본값을 지정한다.
    count = 0 
    # 점수를 한개씩 불러온다
    for i in scores:
        # 불러온 점수가 60점 이상인지 확인한다.
        if i >= 60:
            # 점수가 60점 이상일때 카운트를 한다.
            count +=1
        
    return count
    # 여기에 코드를 작성합니다.


# 아래의 코드는 수정하지 않습니다.
if __name__ == '__main__':
    scores_json = open('problem01_data.json', encoding='UTF8')
    scores = json.load(scores_json)
    print(over(scores)) 
    # => 3
```



># problem 02
>
>음식점 샘플 정보는 json 형식으로 저장되어 있고 이를 불러오기 위한 코드는 미리 작성해 두었다.
>
>음식점 데이터 problem02_data.json의 데이터는 다음과 같이 구성되어있다고 할때 아래의 문제를 해결하시오.
>
>key: id, user_rating, name, menus, location
>
>정보: 각 데이터의 고유값, 유저평점, 상호명, 메뉴리스트, 주소

## problem02_01.py

음식점에서 판매하는 메뉴의 개수를 반환하는 함수 menu_count를 완성하시오.

```python
import json


def menu_count(restorant):

    # 기본값을 지정한다.
    count =0 
    # 메뉴 리스트를 하나씩 불러온다.
    for i in restorant.get('menus'):
        # 메뉴가 하나씩 들어올때 마다 카운트 한다.
        count += 1
        
    return count
    # 여기에 코드를 작성합니다.
    

# 아래의 코드는 수정하지 않습니다.
if __name__ == '__main__':
    restorant_json = open('problem02_data.json', encoding='UTF8')
    restorant = json.load(restorant_json)
    print(menu_count(restorant)) 
    # => 4
```



># problem 03
>
>샘플 정보는 json형식으로 저장되어있고 이를 불러오기 위한 코드는 미리 작성해두었다.
>
>기온데이터 problem02_data.jsom의 데이터는 다음과 같이 구성되어 있다고 할 때 아래의 문제를 해결하시오.



## problem03_01.py

날짜별로 구성되어 있는 데이터를 아래와 값이 maximum, minimum으로 구성한 딕셔너리로 구성하여 반환하는 함수 turn를 완성하시오.

{ 

​    'maximum': [9, 9, 11, 11, 8, 7, -4],

​    'minimum':[3, 0 , -3, 1, -3, -3, -12]

}

```python
import json


def turn(temperatures):
    # 딕셔너리 기본값 지정
    minmax = {}
    # 리스트 기본값지정
    max = []
    min = []
    # 값을 리스트형태로 하나씩 받아온다
    for i in temperatures:
        # max는 max리스트에 min은 min리스트에 추가한다.
        max.append(i[0])
        min.append(i[1])
    # 딕셔너리에 리스트를 추가한다.
    minmax['maximum'] = max
    minmax['minimum'] = min
    # 딕셔너리를 리턴한다.
    return minmax
    # 여기에 코드를 작성합니다.


# 아래의 코드는 수정하지 않습니다.
if __name__ == '__main__':
    temperatures_json = open('problem03_data.json', encoding='UTF8')
    temperatures = json.load(temperatures_json)
    print(turn(temperatures)) 
    # =>
    # {
    #     'maximum': [9, 9, 11, 11, 8, 7, -4], 
    #     'minimum': [3, 0, -3, 1, -3, -3, -12]
    # }
```

