import sys
sys.stdin = open('쉬운 거스름돈.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    N = int(input())

    N_50000 = N // 50000
    N = N - N_50000*50000


    N_10000 = N // 10000
    N = N - N_10000*10000

    N_5000 = N // 5000
    N = N - N_5000*5000

    N_1000 = N // 1000
    N = N - N_1000*1000

    N_500 = N // 500
    N = N - N_500*500

    N_100 = N // 100
    N = N - N_100*100

    N_50 = N // 50
    N = N - N_50*50

    N_10 = N // 10
    N = N - N_10*10
    print('#{}'.format(tc))
    print('{} {} {} {} {} {} {} {}'.format(N_50000,N_10000,N_5000,N_1000,N_500,N_100,N_50,N_10))