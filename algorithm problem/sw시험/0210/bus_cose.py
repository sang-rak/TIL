import sys
sys.stdin = open("bus_cose.txt", "r")
T = int(input())
for tc in range(1, T+1):
    # 최대
    N = int(input()) # 테스트 수
    bus_stop = [0] * 5001
    total_list = []
    for n in range(N):
        arr = list(map(int, input().split()))
        arr_A = arr[0]
        arr_B = arr[1]
        for i in range(arr_A, arr_B+1):
            bus_stop[i] += 1

    P = int(input())

    total = 0
    print(f"#{tc}".format(tc), end=' ')
    for j in range(P):
        number = int(input())

        total = bus_stop[number]

        print(f"{total}".format(total), end=' ')