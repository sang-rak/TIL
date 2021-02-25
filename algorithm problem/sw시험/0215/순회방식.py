arr = [[1,2,3,4],
       [5,6,7,8],
       [10,11,12,13]
       ]

N = 3 # 행의 길이
M = 4 # 열의 길이

len(arr)
len(arr[0])
# 행 우선순회 방식
for i in range(N):
    for j in range(M):
        print(arr[i][j])
# 행 우선순회를 역으로 돌아보자
for i in range(N):
    for j in range(M-1,-1,-1):
        print(arr[i][j], end=" ")

# 열 우선순회 방식
for j in range(M):
    for i in range(N):
        print(arr[i][j])
# 지그재그
for i in range(M):
    for j in range(N):
        print([i][j + (M-1-2*j) * ( i % 2)])
    print()
print()

for i in range