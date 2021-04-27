import sys
sys.stdin = open('이진 탐색.txt', 'r')


def binarySearch(arr, key):
    low, high = 0, len(arr) - 1
    dir = -1  # 방향 미정, 0: 왼쪽, 1: 오른쪽

    while low <= high:
        mid = (low + high) // 2
        # 일치
        if key == arr[mid]:
            return 1
        # 왼쪽방향
        elif key < arr[mid]:
            if dir == 0:
                return 0
            else:
                high = mid - 1
                dir = 0  # 방향 설정
        # 오른쪾 방향
        else:
            # 오른쪽 두번
            if dir == 1:
                return 0
            else:
                low = mid + 1
                dir = 1   # 방향 설정

    return 0


T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    A = sorted(list(map(int, input().split())))
    B = list(map(int, input().split()))

    cnt = 0
    for i in range(M):
        cnt += binarySearch(A, B[i])



    print('#{} {}'.format(tc, cnt))