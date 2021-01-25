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