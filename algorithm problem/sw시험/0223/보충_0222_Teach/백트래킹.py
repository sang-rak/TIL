# 위 아래 오른쪽 왼쪽 대각
dr = [-1, 1, 0, 0, -1, -1, 1, 1]
dc = [0, 0, 1, -1, -1, 1, -1, 1]

arr = [[0] * 10 for _ in range(10)]

# 기준점의 상하좌우
r, c = 0, 0  # (5, 5)
arr[r][c] = 8 # 기준점 8 전환

# 8방향
for i in range(8):
    nr = r + dr[i]
    nc = c + dc[i]

    # 새로운 인덱스를 생성하면 조사
    while 0 <= nr < 10 and 0 <= nc < 10:
        arr[nr][nc] = i+1
        nr = nr + dr[i]
        nc = nc + dc[i]

for lst in arr:
    print(*lst)