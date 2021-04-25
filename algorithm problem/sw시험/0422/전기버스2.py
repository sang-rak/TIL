import sys
sys.stdin = open('전기버스2.txt', 'r')


def dfs(n, k, energy, cnt): # N, k: 정류장 번호, 남은 용량, 교체횟수
    global ans

    if k == n:  # 종점에 도착한 경우
        if ans > cnt:
            ans = cnt
    else:
        # 교체하지 않고 통과
        if energy > 0:
            dfs(n, k+1, energy-1, cnt)
        # 교체하고 통과
        if ans > cnt:  # 가지치기
            dfs(n, k+1, arr[k] - 1, cnt + 1)



T = int(input())

for tc in range(1, T+1):
    arr = list(map(int, input().split())) # 0:N, 충전소 N-1
    ans = 987654321
    dfs(arr[0], 2, arr[1]-1, 0)


    print("#{} {}".format(tc, ans))