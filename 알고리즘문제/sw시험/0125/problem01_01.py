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