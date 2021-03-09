import sys
sys.stdin = open("피자 굽기.txt", "r")

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    Ci = list(map(int, input().split()))
    # 화덕에 넣기
    Ci_in = []

    for i in range(N):
        Ci_in.append([Ci[i], i])

    cnt = 0
    while len(Ci_in) != 1:

        # 화덕 돌리기
        Ci_in[0][0] //= 2

        if Ci_in[0][0] <= 0:
            # 피자개수보다 작으면 더 넣을게 있다는 뜻
            if N + cnt < M:
                Ci_in.pop(0)
                Ci_in.append([Ci[N+cnt], N+cnt])
                cnt += 1
            # 넣을게 없으면 접시만 돌려놓기
            else:
                Ci_in.pop(0)
        else: Ci_in.append(Ci_in.pop(0))
    print('#{} {}'.format(tc, Ci_in[0][1]+1))