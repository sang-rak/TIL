import sys
sys.stdin = open('그룹 나누기.txt', 'r')


def find_set(x):
    if parent[x] == x:
        return x
    else:
        return find_set(parent[x])

def make_set(x):
    parent[x] = x


# 작은 번호로 전환
def union_min(x, y):
    a = find_set(x)
    b = find_set(y)

    if a > b:
        parent[a] = b
    else:
        parent[b] = a


T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    votes = list(map(int, input().split()))
    parent = [0] * (N + 1)


    for i in range(N+1):
        make_set(i)


    for i in range(M):
        union_min(votes[i * 2], votes[i * 2 + 1])

    answer = set()

    for i in parent:
        answer.add(find_set(i))

    print("#{} {}".format(tc, len(answer)-1))
