A = [0, 4, 1, 3, 1, 2, 4, 1]  # 원본
B = [0] * len(A)              # 결과
C = [0] * 5

def countingsort(A, B, C):
    # 카운팅
    for i in range(len(B)):
        C[A[i]] += 1  # 원본의 값을 C의 인덱스로 사용해서 증가
    # 누적
    for i in range(1, len(C)):
        C[i] = C[i] + C[i-1]

    # 자기자리 찾기
    for i in range(len(A)-1, -1, -1):
        B[C[A[i]] - 1] = A[i]
        C[A[i]] -= 1

countingsort(A, B, C)
print(B)















