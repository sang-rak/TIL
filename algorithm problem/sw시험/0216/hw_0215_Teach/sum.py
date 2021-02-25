
import sys

sys.stdin = open("input.txt", "r")

T = 10
N = 100
for tc in range(1, T+1):
    no = int(input())
    arr = [list(map(int, input().split())) for _ in range(N)]
    # print(arr)

    max_value = 0
    # 행우선
    for i in range(N):
        sum_value = 0
        for j in range(N):
            sum_value += arr[i][j]
        if max_value < sum_value:
            max_value = sum_value

    # 열우선
    for i in range(N):
        sum_value = 0
        for j in range(N):
            sum_value += arr[j][i]
        if max_value < sum_value:
            max_value = sum_value
    # 대각선 \
    sum_value = 0
    for i in range(N):
        sum_value += arr[i][i]
        if max_value < sum_value:
            max_value = sum_value
    # print(sum_value)

    # 대각선 /
    sum_value = 0
    for i in range(N):
        sum_value += arr[i][N-1-i]

        if max_value < sum_value:
            max_value = sum_value
    # print(sum_value)
    print("#{} {}".format(no, max_value))