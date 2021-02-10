import sys
sys.stdin = open("flatten_input.txt", "r")
T = 10

# # 최저 높이의 상자 인덱스 위치반환
# def min_search():
#     #초기화
#     min_value = 101
#     min_idx = -1
#
#     for i in range(len(box)):
#         if box[i] < min_value:
#             min_value = box[i]
#             min_idx = i
#     return min_idx
#
# # 최고 높이의 상자 인덱스 위치 반환
# def max_search():
#     max_value = 0
#     max_idx = -1
#
#     for i in range(len(box)):
#         if box[i] > max_value:
#             max_value = box[i]
#             max_idx = i
#     return max_idx
#
#
# T = 10
# for tc in range(1, T + 1):
#     N = int(input())
#     box = list(map(int, input().split()))
#
#     for i in range(N):
#         # 최고 높이 상자 한칸 내리기
#         box[max_search()] -= 1
#         # 최고 높이 상자 한칸 올리기
#         box[min_search()] += 1
#
#     print("#{} {}".format(tc, box[max_search()]-box[min_search()]))
#
# def BubbleSort(arr):
#     for i in range(len(arr)-1, 0, -1): # 4
#         for j in range(i):  # 4 3 2 1
#             if arr[j] > arr[j+1]:
#                 arr[j], arr[j+1] = arr[j+1], arr[j]
#
#
# for tc in range(1, 11):
#     N = int(input())
#     box = list(map(int, input().split()))
#
#     for i in range(N):
#         buble_sort(box)
#         box[0] += 1
#         box[-1] -= 1
#     BubbleSort(box)
#
#     print("{} {}".format(tc, box[-1]-box[0]))



##################### 제일 하고 싶은 코드 양쪽에서 뺌
for tc in range(1, 11):
    N = int(input())
    box = list(map(int, input().split()))

    # 높이 카운트
    h_cnt = [0] * 101

    # 초기화
    min_value = 100
    max_value = 1

    # 박스의 높이를 카운트하면서 최고점과 최저점을 찾아보자.
    for i in range(100):
        h_cnt[box[i]] += 1
        if box[i] > max_value:
            max_value = box[i]
        if box[i] < min_value:
            min_value = box[i]

    while N > 0 and min_value < max_value-1:
        # 상자 옮기기
        h_cnt[min_value] -= 1
        h_cnt[min_value + 1] += 1

        h_cnt[max_value] -= 1
        h_cnt[max_value - 1] += 1

        if h_cnt[min_value] == 0:
            min_value += 1
        if h_cnt[max_value] == 0:
            max_value -= 1

        # 덤프 줄이기
        N -= 1

    print("#{} {}".format(tc, max_value-min_value))