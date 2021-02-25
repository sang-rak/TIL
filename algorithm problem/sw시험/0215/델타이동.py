arr = [[1,2,3],
       [2,3,4],
       [7,8,9]]

r = 1
c = 1
N = 3
# 상하좌우
dr = [-1, 1, 0, 0]
dc = [0, 0, -1, 1]
drc = [[-1, 0], [1, 0], [0, -1], [0, 1]]

for i in range(4):
    nr = r + dr[i]
    nc = c + dc[i]

    if nr < 0 or nr >= N or nc < 0 or nc >= N: continue


    if 0<= nr < N and 0 <= nc <N:
        print(arr[nr][nc])