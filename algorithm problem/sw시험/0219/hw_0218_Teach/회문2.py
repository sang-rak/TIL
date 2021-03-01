import sys
sys.stdin = open("회문2_input.txt", "r")

T = 10

def my_find(M):
    # 전체크기가 N 이다
    for i in range(N):
        # 부분 문자열의 시작점
        for j in range(N-M+1):
            # 스왑을 이용한 회문검사
            for k in range(M//2):
                # 앞뒤검사
                if words[i][j+k] != words[i][j+M-1-k]:
                    break
                elif k == M//2 - 1:
                    return M
            # 세로 검사
            for k in range(M//2):
                if words[j+k][i] != words[j+M-1-k][i]:


for tc in range(1, T+1):
    tc_num = int(input())

    N =100
    words = [input() for i in range(N)]

    # 가장 길이가 긴 회문rjatk흘 한다.
    for i in range(N, 0, -1):
        ans = my_find(i)

        if ans != 0:
            break
    print("# {} {}".format(tc_num, tc))

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
