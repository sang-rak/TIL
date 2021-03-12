import sys
sys.stdin = open('등산로 조성.txt','r')

T = int(input())



for tc in range(1,T+1):
    N, M = map(int, input().split())

    arr = [list(map(int, input().split())) for _ in range(N)]
    print(N, M)
    print(arr)

    # 가장 높은 봉오리 찾기
    for i in range(N):
        for j in range(N):

    # 백트래킹
    # 
    # 지나온길은 체크