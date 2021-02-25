N, M = map(int, input().split())
arr = [0]*N

# for i in range(N):
#     arr[i] = (list(map(int, input().split())))
#
arr = [list(map(int, input().split())) for _ in range(N)]

for i in arr:
    print(i)