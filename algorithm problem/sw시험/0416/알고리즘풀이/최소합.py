# dfs를 이용하는 방법
import time
start_time = time.time()
import sys
sys.stdin = open("최소합.txt", "r")
T = int(input())

def dfs(x, y, dist):
    global ans
    ############################
    if ans <= dist:                            # 목적지 도착전에 이미 다른 경로의 최소값 이상인 경우
      return                                     # 이동을 중단하고 다른 경로로 가면 시간을 줄일 수 있다.
    ############################

    if x == N-1 and y == N-1:                   # 목적지 도착
        if ans > dist :
            ans = dist
    else:
        if x+1 < N:                                   # 아래로 이동 가능한지 확인
            dfs(x+1, y, dist+arr[x+1][y])
        if y+1 < N:                                   # 오른쪽으로 이동 가능한지 확인
            dfs(x, y+1, dist+arr[x][y+1])


for tc in range(1, T+1):
    N = int(input())
    arr = [list(map(int,input().split())) for i in range(N)]
    ans = 987654321                       # 최소값 초기화
    dfs(0, 0, arr[0][0])
    print('#{} {}'.format(tc, ans))
print(time.time() - start_time, 'seconds')
