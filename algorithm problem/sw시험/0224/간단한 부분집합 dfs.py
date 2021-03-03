#dfs

#  { 1, 2, 3} 부분집합
arr = [1, 2, 3]
N = len(arr)
A = [0] * N  # 원소의 포함 여부를 저장

def powerset(n, k): # n: 원소의 개수, k: 현재 depth
    if n == k:  # 기저 상황(멈추는 부분)
        # 솔루션 구하기
        print(A)
        
        # 부분집합선택
        for i in range(n):
            if A[i]:
                print(arr[i], end=' ')
                
        
    else:   # 호출하는 부분
        A[k] = 1
        powerset(n, k+1)
        A[k] = 0
        powerset(n, k+1)


powerset(N, 0)  # depth