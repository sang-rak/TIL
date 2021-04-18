import time
start_time = time.time()

import sys
sys.stdin = open('전자카트.txt', 'r')

def calc(n, k, cursum):
    global ans
    cursum += dist[order[N - 1]][order[N]]  # cursum  0 -> 1 -> 2 까지만 포함됨
    if ans > cursum: ans = cursum


def perm(n, k, cursum):
    if ans < cursum: return

    if n == k:
        calc(n, k, cursum)
    else:
        for i in range(1, n):
            if visited[i]: continue
            order[k] = i
            visited[i] = True
            perm(n, k + 1, cursum + dist[order[k - 1]][order[k]])
            visited[i] = False


def perm2(n, k, cursum):
    if ans < cursum: return

    if n == k:
        calc(n, k, cursum)
    else:
        for i in range(n):  # 0번째 사용 안 함으로 1씩 더해야
            if visited[i + 1]: continue
            order[k + 1] = i + 1
            visited[i + 1] = True
            perm2(n, k + 1, cursum + dist[order[k]][order[k + 1]])
            visited[i + 1] = False


T = int(input())
for tc in range(T):
    ans = 987654321
    N = int(input())  # 3
    order = [0] * N + [0]  # 0 1 2 0 으로 저장(출발, 도착 고정)
    visited = [0] * N

    dist = [list(map(int, input().split())) for _ in range(N)]

    # perm2(N - 1, 0, 0)  # N - 1 : 0번 인덱스 제외
    perm(N, 1, 0)  # k = 0 인덱스 제외
    print('#{} {}'.format(tc + 1, ans))

print(time.time() - start_time, 'seconds')
