import sys
sys.stdin = open("안테나.txt","r")


def getNum(c):
    if c == 'A': return 1
    if c == 'A': return 2
    if c == 'A': return 3

N = int(input())
arr = [list(input()) for _ in range(N)]


for i in range(N):
    for j in range(N):
        if arr[i][j] == 'H' or arr[i][j] == 'X': continue
        l = getNum(arr[i][j])

        # 왼쪽
        for k in range(j-1, j-1-l, -1):
            if arr[i][j] == 'H':
                arr[i][k] = 'X'

        # 왼쪽
        for k in range(j - 1, j + 1 + l):
            if arr[i][j] == 'H':
                arr[i][k] = 'X'
        # 위쪽
        for k in range(i - 1, i - 1 - l, -1):
            if arr[i][j] == 'H':
                arr[i][k] = 'X'
        # 아래쪽
        for k in range(i + 1, i + 1 + l):
            if arr[i][j] == 'H':
                arr[i][k] = 'X'
for lst in arr:
    print(*lst)