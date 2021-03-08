django-admin startproject 프로젝트이름 : 프로젝트 생성



python manage.py startapp articles :  settings.py 에 articles 추가

python manage.py runserver : 프로젝트 안에서 구동

**- 수정 흐름**

1. urls.py

   - from articles import views 

   - path('index/', views.index),

   

2. views.py

   - def index(request):

       pass

3. template

4. 

5. models.py



**- 규칙**

1. app 이름은 복수형
2. app 생성(startapp) 후 등록



base.html에 전체에 해당하는 코드를 넣고

block을통해 상속받아 사용한다.