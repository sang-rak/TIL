import sys
sys.stdin = open('동철이의 일 분배.txt', 'r')


def perm(n, k, curP):
    global ans
    if curP <= ans: return
    if k == n:
        if ans < curP:
            ans = curP
            return

    else:
        for i in range(n):
            if visited[i]: continue
            visited[i] = 1
            order[k] = i
            perm(n, k+1, curP * (arr[k][i]/100))  # arr[k][order[k]]
            visited[i] = 0


T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    visited = [0] * N
    order = [0] * N
    ans = 0
    perm(N, 0, 1)
    # print("#{} {}".format(tc, ans * 100))
    print("#%d %.6f" % (tc, ans * 100))