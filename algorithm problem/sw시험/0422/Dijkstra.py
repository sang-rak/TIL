# 다익스트라 + 인접리스트
'''
서울(0), 천안(1), 원주(2), 논산(3), 대전(4),
대구(5), 강릉(6), 광주(7), 부산(8), 포항(9)
'''
'''
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
'''
[0, 12, 15, 16, 19, 22, 36, 29, 31, 36]
'''
def dijkstra(start):
    # 시작점 0으로 세팅
    u = start     # 시작점을 0으로 설정
    D[u] = 0

    # 정점을 V개 선택
    for i in range(V):
        # 가중치가 최소인 정점을 찾기
        min = 987654321
        for v in range(V):
            if visited[v] == 0 and min > D[v]:
                min = D[v]
                u = v  # 최고값

        # 방문처리
        visited[u] = 1

        # 인접한 정점들의 가중치 업데이트
        for v in range(V):  #u정점의 인접한 v정점들
            if adj[u][v] != 0 and visited[v] == 0 and D[v] > D[u] + adj[u][v]:
                D[v] = D[u] + adj[u][v]

V, E = map(int, input().split())
adj = [[0] * V for _ in range(V)]
#  초기화 작업
D = [987654321] * V     # 가중치를 모두 무한대
PI = list(range(V))           # 내부모는 나다.
visited = [0] * V       # 방문체크

#입력
for i in range(E):
    edge = list(map(int, input().split()))
    adj[edge[0]][edge[1]] = edge[2]     #방향성없음
    adj[edge[1]][edge[0]] = edge[2]  # 방향성있음

dijkstra(0)

print(D)   # MST의 간선 합
