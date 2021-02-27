import sys
sys.stdin = open("회문2_input.txt", "r")

T = 10
def call_me(M, N):
        # 가로에서 확인되면 세로는 안하게
        count = 0
        # 가로 확인
        for j in range(N):
            for k in range(N-M+1):
                cnt = 0
                for t in range(M):
                    if tx_list[j][t+k] == tx_list[j][M+k-t-1]:
                        cnt += 1

                if cnt == M:
                    result = M
                    count += 1
                    return result
        if count == 0:
            # 세로 확인
            for j in range(N):
                for k in range(N-M+1):
                    cnt = 0
                    for t in range(M):
                        if tx_list[t+k][j] == tx_list[M+k-t-1][j]:
                            cnt += 1

                    if cnt == M:
                        for i in range(M):
                            result = M
                            return result

for tc in range(1, T+1):
    # 정답 리스트
    goal_list = []
    _ = input()
    # array 형식 변환
    tx_list = []
    N = 100
    for i in range(N):
        N_str_list = list(map(str, input().split()))
        tx_list += N_str_list

    for M in range(100):
        try:
            if call_me(M, N)>=0:
                result = call_me(M, N)
        except:
            continue
    print("#{} {}".format(tc, result), end=" ")
    print()


#1 18
#2 17
#3 17
#4 20
#5 18
#6 21
#7 18
#8 18
#9 17
#10 18
