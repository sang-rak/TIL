import sys
sys.stdin = open("어디에 단어가 들어갈 수 있을까.txt",'r')

T = int(input())

'''
for tc in range(1, T+1):
    # N: 2차원 리스트 크기, K: 내가 원하는 길이
    N, K = map(int, input().split())

    #리스트 내포 방식을 활용한 입력
    puzzle = [list(map(int, input().split())) for _ in range(N)]

    ans = 0

    for i in range(N):
        cnt = 0
        # 행을 검사
        for j in range(N):
            if puzzle[i][j] == 1:
                cnt += 1
            if puzzle[i][j] == 0 or j == N-1:
                # 벽을 만났을 때 그동안 쌇아온 cnt 값이 k이면 들어갈 수 있다.
                if cnt == K:
                    ans += 1
                cnt = 0

        # 열을 검사
        for j in range(N):
            if puzzle[j][i] == 1:
                cnt += 1
            if puzzle[j][i] == 0 or j == N-1:
                if cnt == K:
                    ans += 1
                cnt = 0
    print("#{} {}".format(tc, ans))
'''

# 가벽을 세우는 방법
for tc in range(1, T+1):
    N, K = map(int, input().split())

    puzzle = [list(map(int, input().split())) + [0] for _ in range(N)]
    puzzle.append([0]*(N+1))
    ans = 0
    for i in puzzle:
        cnt = 0
        # 벽을 한칸 더 둘렀기 때문에 증가
        for j in range(N+1):
            if puzzle[i][j]:
                cnt += 1

            else:
                if cnt == K:
                    ans += 1
                cnt = 0

        # 열 우선 순회
        for j in range(N+1):
            if puzzle[j][i]:
                cnt += 1
            else:
                if cnt == K:
                    ans += 1
                cnt = 0
    print("#{} {}".format(tc, ans))



    # 1 2
    # 2 6
    # 3 6
    # 4 0
    # 5 14
    # 6 2
    # 7 45
    # 8 0
    # 9 98
    # 10 7