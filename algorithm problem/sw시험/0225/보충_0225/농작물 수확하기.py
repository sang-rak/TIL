import sys
sys.stdin = open("농작물 수확하기.txt", "r")

T = int(input())

for tc in range(1, T+1):

    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]

    mid = N//2
    s = e = mid
    ans = 0
    for i in range(N):

        for j in range(s, e+1):
            ans += arr[i][j]
        if i < mid:
            s, e = s - 1, e + 1
        else:
            s, e = s + 1, e - 1

    print(ans)