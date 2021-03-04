import sys
sys.stdin = open("진기의 최고급 붕어빵.txt","r")

T = int(input())

def bread_count(N,M,K,tc):
    cycle_time = list(map(int, input().split()))
    cycle_time = sorted(cycle_time, reverse=True)
    bread_cnt = 0  # 남은 빵 개수
    time = 0  # 현재 시간
    cnt = 0  # 온 사람 카운트

    while cnt != N:

        # work 시간이 되면 뿡어빵 생성
        time += M
        bread_cnt += K

        # 손님이 오면 빵이 줄어든다
        for i in range(N):

            if cycle_time[i] < M:
                return 'Impossible'

            if time <= cycle_time[i] < time + M:
                bread_cnt -= 1
                cnt += 1

        if bread_cnt < 0:
            return 'Impossible'

    return 'Possible'



for tc in range(1, T+1):

    N, M, K = map(int, input().split()) # N: 사람 M: 로트 작업 시간 K: 로트당 붕어빵 생산 수
    result = bread_count(N, M, K, tc)
    print('#{} {}'.format(tc,result))