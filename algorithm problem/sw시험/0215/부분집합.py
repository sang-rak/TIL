# bit = [0,0,0,0]
#
# for i in range(2):
#     bit[0] = i
#     for j in range(2):
#         bit[1] = j
#         for k in range(2):
#             bit[2] = k
#             for l in range(2):
#                 bit[3] = l
#                 print(*bit)
#

'''
-3 3 -9 6 7 -6 1 5 4 -2
'''
# 완전검색
# arr = [1, 2, 3]
arr = list(map(int, input().split()))
n = len(arr)
cnt = 0
for i in range(1, 1 << n): # 2^2 ^(bit연산자) 0~7
    sum = 0
    for j in range(n): # 0~2
        if i & (1 << j): # 0이아닌경우
            # print(arr[j], end=" ")
            sum += arr[j]
    if sum == 0:
        cnt += 1
        for j in range(n):  # 0~2
            if i & (1 << j):  # 0이아닌경우
                print(arr[j], end=" ")

        print()
print("cnt=",cnt)