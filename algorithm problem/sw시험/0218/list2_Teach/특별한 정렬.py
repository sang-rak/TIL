'''
3
10
1 2 3 4 5 6 7 8 9 10
10
67 39 16 49 60 28 8 85 89 11
20
3 69 21 46 43 60 62 97 64 30 17 88 18 98 71 75 59 36 9 26
'''
import sys
sys.stdin = open("특별한 정렬_input.txt", "r")

def selection(a):
    for i in range(10):
        idx = i  #최대값 또는 최소값의 인덱스
        if i % 2 == 0:
            # 최대 인덱스
            for j in range(i + 1, N):
                if a[idx] < arr[j]:
                    idx = j
        else:
            # 최소값
            for j in range(i + 1, N):
                if a[idx] > arr[j]:
                    idx = j
        # 교환
        a[i], a[idx] = a[idx], a[i]


T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = list(map(int, input().split()))

    selection(arr)
    print("#{}".format(tc), end=" ")
    for i in range(10):
        print(arr[i], end=" ")
    print()


# 1 10 1 9 2 8 3 7 4 6 5
# 2 89 8 85 11 67 16 60 28 49 39
# 3 98 3 97 9 88 17 75 18 71 21
