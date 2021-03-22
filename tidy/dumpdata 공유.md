# dumpdata 공유

json 데이터 저장

python manage.py dumpdata --indent 4 movies.movie > movies.json

fixtures\앱이름 폴더에  json 저장



### 받는사람

python manage.py migrate

python manage.py loaddata 앱이름/json이름.json

