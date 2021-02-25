
import sys

sys.stdin = open("input.txt", "r")

T = 10
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(T):
    # ///////////////////////////////////////////////////////////////////////////////////
    a = int(input())
    arr_list = []

    for i in range(1, 101):
        arr = list(map(int, input().split()))
        # 가로의 최대 합
        total = 0
        for j in range(len(arr)):
            total += arr[j]

        if i == 1:
            total_big = total
        else:
            if total_big < total:
                total_big = total

        arr_list.append(arr)

    # 세로의 최대 합
    for k in range(100):
        total = 0
        for n in range(100):
            total += arr_list[n][k]
        if total_big < total:
            total_big = total

    # 선행 대각의 최대 합
    total = 0
    for k in range(100):
        for n in range(100):
            if n == k:
                total += arr_list[n][k]
    if total_big < total:
        total_big = total
    # 역행 대각의 최대 합
    total = 0
    for k in range(100):
        for n in range(100):
            if n == 100-k:
                total += arr_list[k][n]
    if total_big < total:
        total_big = total

    print(f'#{a}',end=' ')
    print(total_big)
    # ///////////////////////////////////////////////////////////////////////////////////
#1 1712
#2 1743
#3 1713
#4 1682
#5 1715
#6 1730
#7 1703
#8 1714
#9 1727
#10 1731
