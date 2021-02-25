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
# print(arr)



def cal_abs(a, b):
    if a - b > 0:
        return a-b
    else:
        return b-a
N = 5
arr = [list(map(int, input().split())) for _ in range(5)]
print(arr)

dx = [0, 1, 0, -1]
dy = [1, 0, -1, 0]

sum_value = 0
# 델타이동 사용
for y in range(N):          # 2차원 배열 순회
    for x in range(N):
        for i in range(4):  # 4방향 탐색
            nx = x + dx[i]
            ny = x + dy[i]
            if nx < 0 or nx >= N: continue
            if ny < 0 or ny >= N: continue
            sum_value += cal_abs(arr[y][x], arr[ny][nx])
        # 방향바꾸는 조건 1. 인덱스 처리 , 2. 앞에 1일때 dir = (dir+1)%4

