import sys
from collections import deque

sys.stdin = open('보급로.txt', 'r')


def dijkstra(queue):
    while len(queue):
        y, x = queue.popleft()
        for dy, dx in move:
            my, mx = y+dy, x+dx

T = int(input())

for i in range(1, T+1):
    N = int(input())
    arr = [list(map(int, input())) for _ in range(N)]
    INF = 987654312
    distance = [[INF for _ in range(N)] for _ in range(N)]

    distance[0][0] = 0
    queue = deque()

    # 시작점
    queue.append((0, 0))

    dijkstra(queue)
    print("#{} {}".format(tc, distance[N-1][N-1]))