# json code

## 딕셔너리가 1개일때

```python
import json
import pprint

# data 폴더 안에 들어있는 music.json 파일을 여는 코드 (인코딩작업)
music_file = open('data/music.json', encoding ='utf-8')
# 불러온 json 파일을 파이썬에서 쓸수있도록 dict로 변환
music_dict = json.load(music_file)

def music_info(music_dict):
	#결과값 반환을 위한 dict
    result = {}
    # 인자로 들어온 music_dict에서 내가 원하는 데이터만 추출
	singer = music_dict['singer']  #1 대괄호 접근법 데이터가 없을때 Error  
    title = music_dict.get('title') #2 get 접근법 데이터가 없을때 None *추천*
    # 결과값 dict에 데이터를 rnwhghk
	result['singer'] = singer
    result['title'] = title
    
    return result
print(music_info(music_dict))

```

## 리스트형

```python
import json
musics_file = open('data/musics.json', encoding = 'utf-8')
musics_list = json.load(music_file)

def music_info():
    for music in musics_list:
        info = {}
        music.get('singer')
        title = music.get('title')
        info['singer'] = singer
        info['title'] = title
        
        # result.append(info)
        result += [info]
    return result 
music_info(musics_list)
```

