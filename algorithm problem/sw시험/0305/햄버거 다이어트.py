import sys
sys.stdin = open('햄버거 다이어트.txt','r')

T = int(input())
# 점수, 칼로리
def powerset(idx):
    global max_point

    # 가지치기
    k = point = 0


    if idx == N:

        for i in range(N):
            if sel[i] == 1:
                k += arr[i][1]
                point += arr[i][0]
        if k < K and max_point < point:
            max_point = point
        return
    else:
        # idx 자리의 원소를 뽑고 간다.
        sel[idx] = 1
        powerset(idx + 1)

        # idx 자리를 안뽑고 간다.
        sel[idx] = 0
        powerset(idx + 1)



for tc in range(1,T+1):
    N, K = map(int, input().split())

    arr = []
    for i in range(N):
        item, money = map(int, input().split())
        arr.append((item, money))

    sel = [0] * N  # a리스트 ( 내가 해당 원소를 뽑았는지 체크)
    max_point = max_kcal = 0
    powerset(0)
    print('#{} {}'.format(tc, max_point))
