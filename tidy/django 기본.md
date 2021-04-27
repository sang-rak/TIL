- 장고 프로젝트 생성

django-admin startproject 프로젝트이름 : 프로젝트 생성

- 장고 앱 생성

python manage.py startapp 앱이름:  settings.py 에 앱 추가

languge code, INSTALLED_APPS 수정

python manage.py runserver : 프로젝트 안에서 구동

## - 수정 흐름

1. urls.py

   - from articles import views 

   - path('index/', views.index),

   

2. views.py

   - def index(request):

   - context라는 변수에 dictionart를 만들어서 넘긴다.

       pass

3. template

   - {{ 변수명(딕셔너리의 키) }}

4. models.py



**- 규칙**

1. app 이름은 복수형
2. app 생성(startapp) 후 등록



base.html에 전체에 해당하는 코드를 넣고

block을통해 상속받아 사용한다.



템플릿 확장

1. templates 폴더를 생성
   - 설정폴더 밑에 생성 또는 프로젝트 폴더 밑에 생성

2. base.html 작성
   - {% block 블럭명 %} {% endblock %} 꼭 추가해준다.

3. settings.py 에 등록
   - Templates 에서 dirs 에 base.html의 경로를 등록해준다.
4. {% extends 'base.html' %}을 각 템플릿 상단에 적어서 사용
   - 템플릿 내용은 {% block 블럭명 %} {% endblock %}사이에 위치

## URL 분리

1. 설정 폴더 urls.py에서 분리를 시작한다.
   - hint: 상단 주석에 방법이 친절히 적혀저 있다
   - url 분리는 applicatiom 단위로 분리
2. include 안에 설정된 위치에 urls.py 생성
3. urls.py 에 내용을 채워넣는다
   - 필요한 함수는 path
   - 리스트는 urlpatterns
4. 사용
   - path(경로, views.함수명)

---

##  URL 사용

1. path함수 세번째 위치에 name='별명' 추가한다.
2. 경로가 필요한 부분(링크)  href="{% url 'dtl' %}"
   - {% url '별명' %}
   - 단점:어플리케이션이 많아지는 경우 동일한 별명이 있을 수 있다.
     - 이러한 경우 어떤 url인지 명확하지 않게 된다.
   - 해결방법: app_name지정
3. app_name지정
   - app_name = 어플리케이션 이름
     - app_name을 설정한 순간부터는 {% url '별명' %} 형식 불가능
   - 사용방법
     - {% url '설정한 app_name:별명' %} 사용해서 구분을 해준다.



## Template namespace

1. 문제의 원인
   - 장고는 INSTALLED_APPS에 등록된 순서대로 template파일 목록을 찾기 때문에 어떤 어플리케이션의 template파일인지 구분 할 수가 없게 된다.
   - 다른 경로로 요청을 했지만 같은 이름의 템플릿 파일의 소속 어플리케이션을 구분할 수 없기 때문에 같은 화면이 보여지는 문제점이 발생한다.
2. 해결방법
   - templates 폴더 내부에 어플리케이션 이름으로 폴더를 하나더 작성한다.



## 연습 첫 번째

- fake google
  - 요청 경로: / pages/fake-google/
  - 보여지는 페이지: 검색어를 입력할 수 있는 페이지가 응답으로 보여진다.
  - 동작
    - 검색어를 입력하고 버튼을 누르면 google결과가 페이지에 보여지게 하자.
    - https://www.google.com/search?q=ssafy 주소로 요청
    - form method: 데이터를 보내는 방식: GET => 쿼리 스트링 형식으로 데이터가 날아간다.
      - url?key=value&key=value





## 연습 2 번째

- 계산기
  - 요청 경로: /pages/calc/숫자/숫자
  - 응답:
    - 페이지에 덧셈결과, 곱셈결과, 나눗셈 결과, 뺄셈 결과 표시.



### WS 로또 당첨 횟수 문제

res = requests.get('https://dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=953').json()





import requests

<<<<<<< HEAD
=======


- admin 아이디 지정

```git
python manage.py createsuperuser
```





- CREATE

1. 인스턴스를 생성하여 데이터를 일일이 넣은 후 저장해주는 방법

   article = Article()

   article .title = ''

   article.content = ''

   **article.save()**

   

2. 인스턴스를 생성하면서 값도 같이 넣은 후 저장해주는 방법

   article = Ariticle(title='', content='')

   **article.save()**



3. 한줄로 작성하는 방법 모델명.objects.create(필드 = 데이터, ....)

   - 리턴 있다. 방금 저장된 인스턴스가 리턴 된다

   Article.objects.create(title='', content='')

   

- READ

1. all() : 전체 정보를 가지고 오는 방법
2. get() : pk일때만 사용
3. filter(): 특정 정보만 가져오는 방법
   - field lookups



- Update
  - 기존의 있는 값을 가져와서
  - 수정 후
  - 다시 save()
- Delete
  - 삭제하려는 인스턴스 값을 가져와서
  - delete()



>>>>>>> d24939cc558199655e06339359d371b4e75bc0c2
