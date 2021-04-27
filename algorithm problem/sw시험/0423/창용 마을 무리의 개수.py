import sys
sys.stdin = open('창용 마을 무리의 개수.txt', 'r')



def find_set(x):
    if x != p[x]:
        p[x] = find_set(p[x])
    return p[x]


T = int(input())

for tc in range(1, T + 1):
    N, M = map(int, input().split())
    p = [i for i in range(N+1)]

    for _ in range(M):
        x, y = map(int, input().split())
        px, py = find_set(x), find_set(y)
        if px != py:
            p[py] = px

    cnt = 0
    for i in range(1, N+1):
        if i == p[i]:
            cnt += 1
    print("#{} {}".format(tc, cnt))