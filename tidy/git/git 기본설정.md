# git 기본설정

$git add helloworld.py

$git commit -m "메시지(변경사항)" (-로 시작하면 보통 short name 옵션)

$git config --global user.name John (--로 시작하면 보통 long name 옵션 arguments(2개)

![image-20210115170444789](네이버 API.assets/image-20210115170444789.png)

$code . : vs 코드 부르기

$git init :폴더부터 하위폴터 모두 git 으로 관리할 예정이다.

$git config --global user.email 이메일주소 : 깃 설정

$git config --global user.name 아이디: 아이디설정

$git status :  현재 git 관리 상태

![image-20210115171541852](네이버 API.assets/image-20210115171541852.png)

$git add . : 모든 파일 등록

$git rm --cached <file> :  잘못 add 했을때 제외

$git commit -m 'first commit' : 관리할 파일목록 저장

git hub에서 repository 만들기

git에서는 붙여넣기 shift + insert 

```
git remote add origin https://github.com/sang-rak/TIL.git
```

git push -u origin master : 푸쉬 하면 로그인 창이 나온다



git chackout : 예전 버전으로 돌아가기 



git clone <code 누르고 https클론주소복사>: 다른 컴퓨터에 git대시를 열때

## 만약 보안이 걸려있다면  

제어판/사용자계정/자격증명 관리자

window 에서 github 보안 풀기

