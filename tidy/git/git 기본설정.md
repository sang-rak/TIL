# git 기본설정

$git add helloworld.py

$git commit -m "메시지(변경사항)" (-로 시작하면 보통 short name 옵션)

$git config --global user.name John (--로 시작하면 보통 long name 옵션 arguments(2개)



$code . : vs 코드 부르기

$git init :폴더부터 하위폴터 모두 git 으로 관리할 예정이다.

$git config --global user.email 이메일주소 : 깃 설정

$git config --global user.name 아이디: 아이디설정

$git status :  현재 git 관리 상태



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



# git 기본설정 2

git init (master 가 보이면 할 필요없다.)



ls 

git status



git add . 

git commit -m '0118 homework'

```
git remote add origin https://lab.ssafy.com/sangrack114/hws.git
```

git push -u origin master (처음 올릴때 -u 추가) 

git push origin master 

touch .gitignore -> vscode로 열기 -> 폴더이름 붙여넣기 -> 맨 뒤에 / 붙이기

.ipyth. git



-------

$ git status
$ git add .
$ git commit -m 'add gitignore'
$ git push origin master
로그인창 로그인후 파일 업로드 확인

git commit -m 'change 영희'



# git 기본설정 3

#지우고 싶을때

cd 그 폴더

ls :확인

ls -a : 모든 파일 확인

git rm --cashed .ipynb_checkpoints -rf

​                            ----------------------------지우고싶은 파일 명

git status : 지워진거 확인

git commit -m 'del 0118/ipynb folder'



웹상에서지우면

git pull origin master 를 해야한다



```
$ git clone git 주소 # 가져올때

$ git init
$ touch .gitignore #gitignore.io검색 -> (python,vs 등 예외 처리)
# *.txt : txt파일 제외
$ touch README
# 내부저장소
$ git status
$ git add .
$ git config --global core.autocrlf true #add 할때 경고창 제거
$ git commit -m 'first'
$ git remote add origin http://원격저장소 주소
$ git push origin master # 이중로그인이라면 자격증명관리자에서 관리가능
$ 

```

