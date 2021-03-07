import sys
sys.stdin = open("그래프 경로.txt")

T = int(input())

# 가지치기 dfs
def dfs(v): # v: 시작정점
    global flag
    if v == e:
        flag = 1
        return 
    # visited 체크: 하고픈 일 해라(출력)
    visited[v] = True
    # print(v, end=" ")

    # 시작정점(v)의 인접한 모든 점정 (w) for 돌리기
    for w in range(1, V+1):
        # 인접정점(w)가 방문하지 않았으면
        if adj[v][w] == 1 and visited[w] == False:
            # 다시 dfs(w) 재귀 호출
            dfs(w)

# return
'''
def dfs(v): # v: 시작정점
    global flag
    if v == e:
        return 1
    # visited 체크: 하고픈 일 해라(출력)
    visited[v] = True
    # print(v, end=" ")

    # 시작정점(v)의 인접한 모든 점정 (w) for 돌리기
    for w in range(1, V+1):
        # 인접정점(w) and 방문하지 않았으면
        if adj[v][w] == 1 and visited[w] == False:
            # 다시 dfs(w) 재귀 호출
            if dfs(w) == 1:
                return 1
    return 0
'''
for tc in range(1, T + 1):
    flag = 0
    V, E = map(int, input().split())  # V: 정점재수, E:간선갯수


    # 0 행렬 그래프 생성: 숫자는 1부터 시작하기 때문에 1개 더 추가
    adj = [[0] * (V+1) for _ in range(V+1)]  # 인접행렬 초기화
    # 지나간 길인지 확인 행렬: 숫자는 1부터 시작하기 때문에 1개 더 추가
    visited = [0] * (V+1)

    # 입력
    for _ in range(E):
        tmp = list(map(int, input().split()))
        u, v = tmp[0], tmp[1]
        adj[u][v] = 1

    # 확인 할 출발 노드 s e
    s, e = map(int, input().split())
    # print(S, G)

    # 행렬 그래프 확인
    # for i in range(V+1):
    #     print("#{} {}".format(i, adj[i]))

    # 함수 실행
    dfs(s)

    # dfs
    '''
    if visited[e] == 1:
        print("#{} 1".format(tc))
    else:
        print("#{} 0".format(tc))
    '''

    # 가지치기
    print("#{} {}".format(tc, flag))
    # print('--------------')