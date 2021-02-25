def selection_sort(a, k):
    # n-1번 최소값 찾기
    for i in range(k):
        #최소 인덱스
        min_idx = i
        for j in range(i+1, N):
            if a [min_idx] > a[j]:
                min_idx = j
        # i값 <-> min 교환
        a[i], a[min_idx] = a[min_idx], a[i]

def selection_sort(a, N):
    # n-1번 최소값 찾기
    for i in range(N-1):
        #최소 인덱스
        min_idx = i
        for j in range(i+1, N):
            if a [min_idx] > a[j]:
                min_idx = j
        # i값 <-> min 교환
        a[i], a[min_idx] = a[min_idx], a[i]

arr = [64, 25, 10, 22, 11]
N = len(arr)
selection_sort(arr, N)
print(arr)