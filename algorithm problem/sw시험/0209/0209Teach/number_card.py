import sys
sys.stdin = open("number_input.txt", "r")


def countingsort(A, B, C):
    # 카운팅
    for i in range(len(B)):
        C[int(A[i])] += 1  # 원본의 값을 C의 인덱스로 사용해서 증가


T = int(input())
for tc in range(1,T+1):
    N = int(input())
    arr = list(map(str, input().split()))

    A = arr[0]
    B = [0] * len(A)  # 결과
    C = [0] * 10

    countingsort(A, B, C)
    
    # 처음 값 지정
    max_cp = C[0]
    max_int = 0
    for j in range(1, len(C)):
        if max_cp <= C[j]:
            max_cp = C[j]
            max_int = j

    print(f"#{tc} {max_int} {max_cp}".format(tc, max_int, max_cp))
