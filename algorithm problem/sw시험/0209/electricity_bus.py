import sys
sys.stdin = open("electricity_input.txt", "r")

T = int(input())
for tc in range(1, T+1):

    # 최대
    K, N, M = list(map(int, input().split())) # K: 최대 이동수, N: N번 정류장까지 이동   ,M:충전기가 있는 정류장 번호 개수
    arr = list(map(int, input().split())) # print(arr) # M:충전기가 있는 정류장 번호

    A = [0] * (N+1)  # 정류장 수 + 1
    count = 0  # 충전 횟수
    load = 0  # 이동 거리
    load_a = 0  # 이동거리 임시 저장

    # 충전기 있는 정류장 표시
    for i in range(len(arr)):
        A[arr[i]] = 1

    # K칸만 지나면 도착하기 한 턴 전에 멈춤
    while load < N-K:

        # K 칸안에 정류장이 있다면 제일 멀리간 정류장 번호
        for j in range(1, K+1):
            if A[load+j] == 1:
                load_a = load+j

        # load 가 여러번 반복되면 정지하면서 0 반환
        if load_a == load:
            count = 0
            break

        load = load_a
        count += 1

    print(f"#{tc} {count}".format(tc, count))
