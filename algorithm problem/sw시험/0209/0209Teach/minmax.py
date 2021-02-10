import sys
sys.stdin = open("min_max.txt", "r")

T = int(input())
# for tc in range(1,T+1):
#     N = int(input())
#     arr = list(map(int, input().split()))
#     # 최대
#     for i in range(len(arr)):
#         if i == 0:
#             data_max = arr[i]
#             data_min = arr[i]
#         else:
#             if data_max < arr[i]:
#                 data_max = arr[i]
#                 if data_min > arr[i]:
#                     data_min = arr[i]
#             else:
#                 if data_min > arr[i]:
#                     data_min = arr[i]
#
#     print(f"#{tc} {data_max-data_min}".format(tc, data_max, data_min))

#