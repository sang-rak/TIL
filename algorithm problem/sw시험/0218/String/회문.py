import sys
sys.stdin = open("회문_input.txt", "r")

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    # 정답 리스트
    goal_list = []

    # array 형식 변환
    tx_list = []
    for i in range(N):
        N_str_list = list(map(str, input().split()))
        tx_list += N_str_list

    # print(tx_list[0][0])

    # 가로 확인
    for j in range(N):
        for k in range(N-M+1):
            cnt = 0
            for t in range(M):
                if tx_list[j][t+k] == tx_list[j][M+k-t-1]:
                    cnt += 1
            if cnt == M:
                print("#{}".format(tc), end=" ")
                print(tx_list[j][k:k+M])

            # 세로 확인
            for k in range(N-M+1):
                cnt = 0
                for t in range(M):
                    if tx_list[t+k][j] == tx_list[M+k-t-1][j]:
                        cnt += 1
                if cnt == M:
                    print("#{}".format(tc), end=" ")

                    for i in range(M):
                        print(tx_list[k+i][j], end="")
                    print()

