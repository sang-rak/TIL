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