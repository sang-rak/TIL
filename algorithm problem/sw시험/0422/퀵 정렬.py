import sys
sys.stdin = open('퀵 정렬.txt', 'r')


def quickSort(arr, left, right):
    if left < right:
        pivot = partition(arr, left, right)
        quickSort(arr, left, pivot-1)
        quickSort(arr, pivot+1, right)


def partition(arr, left, right):
    pivot = arr[left]
    i, j = left, right
    while i < j:
        while arr[i] <= pivot and i < right:
            i += 1
        while arr[j] >= pivot and j > left:
            j -= 1
        if i < j:
            arr[i], arr[j] = arr[j], arr[i]
    arr[left], arr[j] = arr[j], arr[left]
    return j



T = int(input())

for tc in range(1, T+1):
    N = int(input())
    A = list(map(int, input().split()))

    quickSort(A, 0, N-1)

    print("#{} {}".format(tc, A[N//2]))