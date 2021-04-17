import sys
sys.stdin = open('전자카트.txt', 'r')

def using(start):
    global sub_result, result, final_result

    if len(sub_result) == N-1:
        for i, j in sub_result:
            result += battery[i][j]

        # 마지막은 제자리로
        result += battery[start][0]
        final_result.append(result)
        result = 0
        return

    for next in range(1, N):
        if not visited[next]:
            sub_result.append((start, next))
            visited[next] = True
            using(next)
            sub_result.remove((start, next))
            visited[next] = False

T = int(input())

for tc in range(1, T+1):
    N = int(input())
    battery = [list(map(int, input().split())) for _ in range(N)]
    visited = [0] * N
    sub_result = []
    result = 0
    final_result = []

    using(0)
    final_result = min(final_result)
    print("#{} {}".format(tc, final_result))

