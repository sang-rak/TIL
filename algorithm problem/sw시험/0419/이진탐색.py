

def binarySearch(arr, n, key):
    low = 0
    high = n - 1
    while low <= high:
        mid = (low + high) // 2

        if key == arr[mid]:  # 탐색 성공
            return mid
        elif key < arr[mid]: # 중간 값보다 작은 경우
            high = mid - 1

        else:                # 중간값보다 작은 경우
            low = mid + 1

    return -1


key = 7
arr = [2, 4, 7, 9, 11, 19, 23]
print(binarySearch(arr, len(arr), key))