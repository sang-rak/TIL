import sys
sys.stdin = open('이진 탐색.txt', 'r')


def binary_sort(a, key):
    l = 0
    r = len(a) - 1
    while l <= r:
        m = (l + r) // 2
        # 같은 경우
        if key == a[m]:
            return 1
        # key 값이 작은경우
        elif key < a[m]:
            if dir == 0:
                return 0
            else:
                dir = 0    # 왼
                r = min - 1

        else:
            if dir == 1:
                return
            else:
                dir = 1     # 오
                l = min + 1

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    arr_N = sorted(list(map(int, input().split())))
    arr_M = list(map(int, input().split()))
    print('#{} {}'.format(tc, binary_sort()))