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