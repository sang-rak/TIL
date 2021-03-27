# branch 관리

- git branch

  branch 확인

- 생성

  git branch <branch name>

- 이동

  git switch <branch name>

- 삭제

  git brach -d <branch name>

- 저장

  git push origin <branch name>

- 병합
  git merge <branch name>

- master가 아닌 branch끼리의 병합
  1. git switch dev
  2. git merge <합칠 branch name>
- git push
  1. git remote add pjt06 주소
  2. git push pjt06 master

병합후 삭제 필수



 feature branches

develop 중복되는것들

release



master



## 프로젝트 가이드

pjt04 파일 생성

1. 기본 PUSH 프로세스
2. 멤버 등록
3. B: git clone <git 주소> pjt06 B
4. git branch <작업이름>
5. git switch <작업이름>
6. git commit 하기
7. git swich master
8. git merge <작업이름>
9. git push origin master



<B>

1. git pull origin master

