'''
11 45 22 81 23 34 99 22 17 8
69 10 30 2 16 8 31 22
1 1 1 1 1 0 0 0 0 0
1 2 3 4 5 6 7 8 9 10
'''

def quicksort(arr, key, n):
    low = 0
    high = n-1

    while low <= high and location == 0:
        mid = low + (high + low) // 2

        if arr[mid] == key:
            return mid
        elif arr[mid] > key:
            high = mid - 1
        else:
            low = mid + 1
    return -1

arr = list(map(int, input().split()))
result = quicksort(arr, 0, len(arr)-1)
print(arr)
print(result)
