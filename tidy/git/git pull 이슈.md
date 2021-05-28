# git pull 이슈



코드작성중 내 코드를 포기하고 상대가 commit  한내역으로 돌아갈 때

git stash && git pull origin main && git stash pop





error: Pulling is not possible because you have unmerged files.

```
git fetch -all
git reset --hard origin/main
git pull origin main
```

