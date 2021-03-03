import sys
sys.stdin = open("안테나.txt","r")

N = int(input())
arr = [list(input()) for _ in range(N)]

for i in range(N):
    for j in range(N):
        if arr[i][j] == 'A':
            if arr[i - 1][j] == 'H':
                arr[i - 1][j] ='X'
            if arr[i + 1][j] == 'H':
                arr[i + 1][j] ='X'
            if arr[i][j+1] == 'H':
                arr[i][j+1] ='X'
            if arr[i][j-1] == 'H':
                arr[i][j-1] ='X'

cnt = 0
for i in range(N):
    for j in range(N):
        if arr[i][j] == 'H':
            cnt += 1
print(cnt)