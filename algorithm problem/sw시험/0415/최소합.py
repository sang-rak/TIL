import sys
sys.stdin = open('최소합.txt', 'r')

dxy = [[0, 1], [1, 0]]

def load(i, j, minsum):
    global min_result
    minsum += arr[i][j]
    # 이미 더 크면 멈춰!
    if minsum >= min_result:
        return

    if i == N-1 and j == N-1:
        min_result = minsum
        return

    for idx in range(2):
        ni, nj = i + dxy[idx][0], j + dxy[idx][1]
        if ni < N and nj < N:
            load(ni, nj,  minsum)

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    use_check = [0 for _ in range(N)]

    min_result = 99999999

    load(0, 0, 0)
    print("#{} {}".format(tc, min_result))

