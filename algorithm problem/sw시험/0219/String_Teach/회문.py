import sys
sys.stdin = open("회문_input.txt", "r")

T = int(input())

def my_reverse(line):
    r_line = []
    for i in range(len(line)-1, -1, -1):
        r_line.append(line[i])

    return r_line



def my_find():
    # 전체 크기가 N
    for i in range(N):
        # 가로 검사
        for j in range(N-M+1):
            tmp = []
            for k in range(M):
                tmp.append(words[i][j+k])
            # 회문 검사
            if tmp == my_reverse(tmp):
                return tmp

        # 세로 검사
        for j in range(N-M+1):
            tmp = []
            for k in range(M):
                tmp.append(words[j+k][i])
            if tmp == my_reverse(tmp):
                return tmp


for tc in range(1, T+1):
    N, M = map(int, input().split())

    words = [list(input()) for _ in range(N)]
    ans = my_find()

    print("#{} {}".format(tc, ''.join(ans)))

'''
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
                    # print(tx_list[j][t], end=' ')
                    # print(tx_list[j][M-t-1])
            if cnt == M:
                print("#{}".format(tc), end=" ")
                print(tx_list[j][k:k+M])

    # 세로 확인
    for j in range(N):
        for k in range(N-M+1):
            cnt = 0
            for t in range(M):
                if tx_list[t+k][j] == tx_list[M+k-t-1][j]:
                    cnt += 1
                    # print(tx_list[j][t], end=' ')
                    # print(tx_list[j][M-t-1])
            if cnt == M:
                print("#{}".format(tc), end=" ")

                for i in range(M):
                    print(tx_list[k+i][j], end="")
                print()

'''