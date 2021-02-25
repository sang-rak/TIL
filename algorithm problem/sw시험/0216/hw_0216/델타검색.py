'''
1 1 1 1 1
1 0 0 0 1
1 0 0 0 1
1 0 0 0 1
1 1 1 1 1

24
'''
# 입력
# arr = [[0 for _ in range(5)] for _ in range(5)]
# for i in range(5):
#     arr[i] = list(map(int, input().split()))

def cal_abs(a, b):
    if a - b > 0:
        return a - b
    else:
        return b - a

#입력
N = 5
arr = [list(map(int, input().split())) for _ in range(5)]
print(arr)

# 상하좌우
dx = [0, 0, -1, 1]
dy = [-1, 1, 0, 0]

sum_value = 0
for y in range(N):           # 2차원배열 순회
    for x in range(N):
        for i in range(4):   # 4방향 탐색
            nx = x + dx[i]
            ny = y + dy[i]
            # if nx < 0 or nx >= N: continue
            # if ny < 0 or ny >= N: continue
            if 0 <= nx < N and 0 <= ny < N:
            #if nx in range(N) and ny in range(N):
                sum_value += cal_abs(arr[y][x], arr[ny][nx])
print(sum_value)
