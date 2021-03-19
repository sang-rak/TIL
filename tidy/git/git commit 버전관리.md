# 왠만하면 사용하지 X 충돌발생 가능성



- add의 반대

git rm --cached a.txt



- commit의 반대





- git 확인

  ```git log
  git log --oneline
  ```

- commit 했던 바로 전까지

  ```
  git commit --amend
  ```

  

- 현재상태 저장후 나가기

  ```
  :wq
  ```

  

# commit 버전관리

```
# 영화보기

1. 영화를 보기 위해 영화관 도착!!
2. 명량 영화표 구매
3. 팝콘구매
4. 상영시간이 될때 까지 웹서핑
5. 댓글읽다가 주인공 사망 스포당함
```

- 과거로 완전히 돌아감 코드도 돌아감

  git reset --hard 86c4d71

- 과거로 완전히 돌아감 코드도 돌아감

  git reset --soft 86c4d71