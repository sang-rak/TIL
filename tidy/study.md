# static 설정: css, img, js

> 개발자가 준비

1. base,html 수정
   - block 설정
2. index.html 수정 
   - block 설정
   - load static
   - href = "{% static %}"
3. CSS파일 생성
4. 서버 재실행



# Media 설정

> 사용자가 업로드

1. settings.py에 MEDIA_RROT, MEDIA_URL 설정
2. upload_to 속성을 정의하여
   - MEDIA_ROOT
   - <form ~~~ enctype="det_part/~~"
   - 저장할때, request.FILES 추가
3. show
   - MEDIA_URL
   - urls.py(설정폴더) media 관련 url 추가,