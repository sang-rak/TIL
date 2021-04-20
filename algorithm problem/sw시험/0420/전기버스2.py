import sys
sys.stdin = open('전기버스2.txt', 'r')


def dfs(num):
    global cnt, result
    
    # 최단거리
    if num >= N:
        if result > cnt:
            result = cnt
        return

    # 가지치기
    if result < cnt:
        return

    start = num
    life = data[start]

    for i in range(start+life, start, -1):
        cnt += 1
        dfs(i)
        cnt -= 1


T = int(input())

for tc in range(1, T+1):
    data = list(map(int, input().split()))

    N = data[0]
    result = 999999999
    cnt = 0

    dfs(1)

    print("#{} {}".format(tc, result-1))