# dfs

#  { 1, 2, 3} 부분집합
arr = list(range(1, 11))

N = len(arr)
A = [0] * N  # 원소의 포함 여부를 저장

def powerset(n, k, cursum):  # n: 원소의 개수, k: 현재 depth
    global cnt
    global total

    # 가지 치기
    if cursum > 10: return
    total += 1
    if n == k:  # 기저 상황(멈추는 부분)
        # 솔루션 구하기
        # 부분집합선택
        if cursum == 10:
            cnt += 1
            for i in range(n):
                if A[i]:
                    print(arr[i], end=' ')

    else:  # 호출하는 부분
        A[k] = 1
        # cursum += arr[k]
        powerset(n, k + 1, cursum + arr[k])
        A[k] = 0
        powerset(n, k + 1, cursum)

cnt = 0
total = 1
powerset(N, 0, 0)  # depth
print(total)