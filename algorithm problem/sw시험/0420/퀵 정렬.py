import sys
sys.stdin = open('퀵 정렬.txt', 'r')


def quick_sort(arr, left, right):
    if left < right:
        p = partition(arr, left, right)
        if p == N//2:
            return
        quick_sort(arr, left, p - 1)
        quick_sort(arr, p + 1, right)


def partition(arr, left, right):
    pivot = (left + right) // 2
    while left < right:
        while arr[left] < arr[pivot] and left < right:
            left += 1
        while arr[right] >= arr[pivot] and left < right:
            right -= 1
        if left < right:
            if left == pivot:
                pivot = right
            arr[left], arr[right] = arr[right], arr[left]
    arr[pivot], arr[right] = arr[right], arr[pivot]
    return right


T = int(input())

for tc in range(1, T+1):
    N = int(input())
    arr = list(map(int, input().split()))

    quick_sort(arr, 0, N-1)

    print("#{} {}".format(tc, arr[N//2]))