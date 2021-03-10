import sys
sys.stdin = open('러시아 국기 같은 깃발.txt','r')

T = int(input())

def perm(idx, sub_sum):
    global ans
    # 유망성 검사 아래의 조건문에 걸리게 되면
    # 이후 작업은 의미가 없음
    if sub_sum > N:
        return

    if idx == 3:
        if sub_sum == N:
            cnt = 0

            st = sel[0]
            st2 = st + sel[1]

            # 흰색 칠하기
            for i in flag[:st]:
                for j in i:
                    if j != 'W':
                        cnt += 1

            # 파란색 칠하기
            for i in flag[st:st2]:
                for j in i:
                    if j != 'B':
                        cnt +=1

            # QKfrks색 칠하기
            for i in flag[st2:]:
                for j in i:
                    if j != 'R':
                        cnt += 1

            if ans > cnt:
                ans = cnt
        return

    # 중복순열 살짝 응용
    for i in range(1, N+1):
        sel[idx] = i
        perm(idx+1, sub_sum+i)

for tc in range(1, T+1):
    N, M = map(int, input().split()) # N개의 줄에는 M개의 문자
    # ‘W’는 흰색, ‘B’는 파란색, ‘R’은 빨간색
    flag = [list(map(str, input())) for _ in range(N)]
    sel = [0] * 3
    ans = 987654321

    # 앞에는 idx, 뒤에는 중간 합
    perm(0, 0)

    print("#{} {}".format(tc, ans))
'''

for tc in range(1, int(input()) + 1):
    N, M = map(int, input().split())

    flag = [input() for _ in range(N)]

    W = [0] * N
    B = [0] * N
    R = [0] * N

    # 행을 보면서 나와 다른 색깔의 개수를 카운팅
    for i in range(N):
        for j in range(M):
            if flag[i][j] != 'W':
                W[i] += 1
            if flag[i][j] != 'B':
                B[i] += 1
            if flag[i][j] != 'R':
                R[i] += 1

    # 누적 시키자
    for i in range(1, N):
        W[i] += W[i-1]
        B[i] += B[i-1]
        R[i] += R[i-1]

    ans = 987645321

    for i in range(N-2):
        for j in range(i+1, N-1):
            w_cnt = W[i]
            b_cnt = B[j] - B[i]
            r_cnt = R[N-1] - R[j]

            if ans > w_cnt + b_cnt + r_cnt:
                ans = w_cnt + b_cnt + r_cnt
    print("#{} {}".format(tc, ans))

'''