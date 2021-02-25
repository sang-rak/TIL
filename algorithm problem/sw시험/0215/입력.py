'''
3 4
1 2 3 4
5 6 7 8
9 10 11 12
'''

# 예시 1
# n, m = map(int, input().split())
# mylist = [0] * n
# for i in range(n):
#     mylist[i] = list(map(int, input().split()))
#
# print(mylist)

# 예시 2
# n, m = map(int, input().split())
# mylist2 = []
# for i in range(n):
#     mylist2.append(list(map(int, input().split())))
#
# print(mylist2)

# 예시 3
n, m = map(int, input().split())
mylist = [list(map(int, input().split())) for _ in range(n)]
print(mylist)

# 2차원 배열 초기화
arr = [[0 for _ in range(4)] for _ in range(3)]
# arr = [[0]*3] * 4 # 이렇게 하면 안된다.
# arr[0][0] = 1 일때 같이 변하기 때문에
arr = [[0] * 4 for _ in range(3)] # 초기화 방법

print(arr)

for i in rangg