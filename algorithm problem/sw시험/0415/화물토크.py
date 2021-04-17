import sys
sys.stdin = open('화물토크.txt', 'r')


T = int(input())

for tc in range(1, T+1):
    N = int(input())

    gant = []
    for _ in range(N):
        s, e = map(int, input().split())
        gant.append([s, e])
    gant.sort(key=lambda e : e[1])
    result = 0
    lot_time = 0  # 작업 끝난 시간

    while gant:
        start, end = gant.pop(0)
        if start >= lot_time:
            result += 1
            lot_time = end
    print('#{} {}'.format(tc, result))