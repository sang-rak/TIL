import sys
sys.stdin = open("별삼각형1.txt", "r")

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())

    print("#{}".format(tc))
    if M == 1:
        for i in range(1, N+1):
            print('*'*i)
    elif M == 2:
        for i in range(N, 0, -1):
            print('*'*i)
    else:
        for j in range(0, N):
            print(' ' * (N-j-1) + '*' + '*' * 2 * j)

