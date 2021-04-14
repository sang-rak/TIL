#  0,120,12,7,76,24,60,121,124,103

arr = [0,0,0,0,0,0,0,1,1,1,  1,0,0,0,0,0,0,1,1,0,  0,0,0,0,0,1,1,1,1,0,
       0,1,1,0,0,0,0,1,1,0,  0,0,0,1,1,1,1,0,0,1,  1,1,1,0,0,1,1,1,1,1,  1,0,0,1,1,0,0,1,1,1]


# 1 2 4 8 16 32 64
# for i in range(0, len(arr), 7):
#     n = 0
#     for j in range(i, i+7):
#         n = n * 2 + arr[j]
#     print(n, end=' ')
# print()

# 마지막이 7보다 작을 때
for i in range(0, len(arr), 7):
    cnt = n = 0
    j = i
    while j < len(arr) and cnt < 7: # j 가 arr 에 끝이 아니고 cnt 7 보다 작을 경우
        n = n * 2 + arr[j]
        cnt += 1
        j += 1
    print(n, end=' ')
print()

# for i in range(len(arr)//7):
#     temp = arr[i*7:i*7+7]
#     result = 0
#     if temp[0] == 1:
#         result += 64
#     if temp[1] == 1:
#         result += 32
#     if temp[2] == 1:
#         result += 16
#     if temp[3] == 1:
#         result += 8
#     if temp[4] == 1:
#         result += 4
#     if temp[5] == 1:
#         result += 2
#     if temp[6] == 1:
#         result += 1
#
#     print(result, end=' ')