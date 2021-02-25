# arr = [10, 15, 2, 19, 6, 14]
#
# for i in range(len(arr)-1):
#     min_idx = i
#
#     for j in range(i+1, len(arr)):
#         if arr[j] < arr[min_idx]:
#             min_idx = j
#
#     arr[i], arr[min_idx] = arr[min_idx], arr[i]
#
# print(arr)

def binary_search(a, key): # arr, key
    start = 0
    end = len(a)-1

    while start <= end:
        mid = (start + end)//2
        # mid 일때
        if a[mid] == key:
            return mid, True
        # key 작을 때
        elif a[mid] > key:
            # end mid보다 하나 아래로
            end = mid-1
        # key 클 때
        # elif a[mid] < key:
        else:
            #start를 mid보다 하나 크게
            start = mid + 1
    return -1, False

arr = [2, 4, 7, 9, 11, 19, 23]
key = 22
print(binary_search(arr, key))