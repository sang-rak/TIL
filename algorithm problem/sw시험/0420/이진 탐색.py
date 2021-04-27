import sys
sys.stdin = open('이진 탐색.txt', 'r')


def f():
    result = 0
    for i in arr_M:
        before = None
        left = 0
        right = N-1
        while left <= right:
            pivot = (left + right) // 2
            if i == arr_N[pivot]:
                result += 1
                break
            elif i < arr_N[pivot]:
                right = pivot - 1
                now = 'left'
            elif i > arr_N[pivot]:
                left = pivot + 1
                now = 'right'
            if before == now:
                break
            before = now
    return result


T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    arr_N = sorted(list(map(int, input().split())))
    arr_M = list(map(int, input().split()))
    print('#{} {}'.format(tc, f()))