

import sys
sys.stdin = open("병합 정렬.txt")

def merge_sort(a):
    global cnt
    # 문제를 절반으로 나누어서 각각을 해결하는 부분
    if len(a) == 1:
        return a
    else:
        mid = len(a) // 2
        left = a[:mid]
        right = a[mid:]

        left = merge_sort(left)
        right = merge_sort(right)

    # 두개의 정렬된 집합을 하나의 집합으로 만들어서 반환
        i = left_idx = right_idx = 0
        # 왼쪽 오른쪽이 모두 존재할 때
        while left_idx < len(left) and right_idx < len(right):
            if left[left_idx] <= right[right_idx]:  # 안정정렬  = 추가
                a[i] = left[left_idx]
                left_idx += 1
            else:
                a[i] = right[right_idx]
                right_idx += 1
            i += 1
        # 왼쪽만 남아 있는 경우
        if left_idx < len(left): # a 뒤에 left_idx 뒤에 모든 원소를 추가해라
            a[i:]= left[left_idx:]
        # 오른쪽만 남아 있는 경우
        if right_idx < len(right):
            a[i:]= right[right_idx:]

        if left[-1] > right[-1]: # 왼쪽 마지막 원소가 큰 경우
            cnt += 1
        return a

T = int(input())
for tc in range(1, T+1):
    N = int(input())
    A = list(map(int, input().split()))
    cnt = 0
    A = merge_sort(A)
    print("#{} {} {}".format(tc, A[N//2], cnt))
