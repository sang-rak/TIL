import sys
sys.stdin = open("별삼각형2.txt", "r")

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())

    print("#{}".format(tc))
    if M == 1:
        for i in range(1, N//2+1):
            print('*'*i)
        for i in range(N//2+1, 0, -1):
            print('*'*i)

    elif M == 2:
        for j in range(0, N//2):
            print(' ' * (N-j-3) + '*' + '*' * j)
        for j in range(N//2+1, 0, -1):
            print(' ' * (N-j-2)  + '*' * j)


    elif M == 3:
        for j in range(N//2, 0, -1):
            print(' ' * (N-j-3) + '*' + '*' * 2 * j)
        for j in range(0, N//2+1):
            print(' ' * (N-j-3) + '*' + '*' * 2 * j)

    else:
        for j in range(N//2+1, 0, -1):
            print(' ' * (N-j-2)  + '*' * j)
        for j in range(0, N//2+1):
            print(' ' * (N-3) + '*' + '*' * j)