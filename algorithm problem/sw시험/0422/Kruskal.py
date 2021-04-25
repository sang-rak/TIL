'''
서울(0), 천안(1), 원주(2), 논산(3), 대전(4),
대구(5), 강릉(6), 광주(7), 부산(8), 포항(9)

10 14
0 1 12
0 2 15
1 3 4
1 4 10
2 5 7
2 6 21
3 4 3
3 7 13
4 5 10
5 8 9
5 9 19
6 9 25
7 8 15
8 9 5
'''

def make_set(x):
    parent[x] = x

def find_set(x):
    if parent[x] == x:
        return x
    else:
        return find_set(parent[x])

def union(x, y):
    parent[find_set(y)] = find_set(x)

def kruskal():
    total = 0
    cnt = 0
    # make set
    for i in range(V):
        make_set(i)

    # 가중치별로 정렬
    edges.sort(key=lambda x: x[2])

    # findset 비교
    for i in range(E):
        if find_set(edges[i][0] != edges[i][1]):
            total += edges[i][2]
            cnt += 1
            union(edges[i][0], edges[i][1])
        if cnt == V-1: break
    return total

V, E = map(int, input().split())
# 간선의 배열  : 시작 끝 가중치
edges = [list(map(int, input().split())) for _ in range(E)]
parent = [0] * V
print(kruskal())
