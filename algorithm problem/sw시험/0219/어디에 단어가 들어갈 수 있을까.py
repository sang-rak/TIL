import sys
sys.stdin = open("어디에 단어가 들어갈 수 있을까.txt")
T = int(input())

for t in range(1, T+1):
    N, K = map(int, input().split())
    tmp = [list(map(int, input().split())) for _ in range(N)]
    ans = 0

    for i in range(N):
        cnt_row = 0
        cnt_col = 0

        for j in range(N):
            # 행
            if tmp[i][j] == 1 :
                cnt_row += 1
            if tmp[i][j] == 0 or j == N-1:
                if cnt_row == K:
                    ans += 1
                cnt_row = 0
            # 열
            if tmp[j][i] == 1 :
                cnt_col += 1
            if tmp[j][i] == 0 or j == N-1:
                if cnt_col == K:
                    ans += 1
                cnt_col = 0
    print("#{} {}".format(t, ans))