# dumpdata 공유

json 데이터 저장

python manage.py dumpdata --indent 4 movies.movie > movies.json

fixtures\앱이름 폴더에  json 저장



### 받는사람

python manage.py migrate

python manage.py loaddata 앱주소/json이름.json



##### dumpdata 생성 4개씩

python manage.py dumpdata --indent 4



- 기본 명령어

**python manage.py dumpdata > xxx.json**





이렇게 명령어를 치면 내 장고가 사용하고 있는 모든 DB의 데이터를 json형식으로 저장을 알아서 해준다.





- 특정 앱의 데이터만 덤프할때

**python manage.py dumpdata appname > xxx.json**





dumpdata 뒤에 파라미터로 앱이름을 적어주면 해당 앱의 내용만 데이터 덤프가 된다.





- 특정 데이터만 빼고 덤프할때

**python manage.py dumpdata --exclude appname > xxx.json**





dumpdata 뒤에 파라미터로 제외할 앱이름을 적어주면 된다.

