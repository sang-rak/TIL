import sys
sys.stdin = open('컨테이너 운반.txt', 'r')


def f(N, M):
    moved = [0]*N
    s = 0
    for i in range(M):
        for j in range(N):
            if moved[j] == 0 and t[i] >= w[j]: # 남은 화물 중 큰 것부터
                s += w[j]
                moved[j] = 1
                break  # 다음 트럭으로
    return s


T = int(input())

for tc in range(1, T+1):
    # 컨테이너 수 N과 트럭 수 M
    N, M = map(int, input().split())

    w = list(map(int, input().split()))
    t = list(map(int, input().split()))
    w.sort(reverse=True)

    print('#{} {}'.format(tc, f(N,M)))
