'''
3
10
1 2 3 4 5 6 7 8 9 10
10
67 39 16 49 60 28 8 85 89 11
20
3 69 21 46 43 60 62 97 64 30 17 88 18 98 71 75 59 36 9 26
'''
import sys

sys.stdin = open("특별한 정렬_input.txt", "r")

def BubbleSort(x_list):
    for i in range(len(x_list) - 1, 0, -1):
        for j in range(i):
            if x_list[j] > x_list[j + 1]:
                x_list[j], x_list[j + 1] = x_list[j + 1], x_list[j]
    return x_list

T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):
    N = int(input())
    list_a = list(map(int, input().split()))
    sort_a = BubbleSort(list_a)
    sort_list = [0]*N
    for k in range(1,N+1):
        if k % 2 == 1:
            sort_list[k-1] = sort_a.pop(-1)

        else:
            sort_list[k-1] = sort_a.pop(0)
    print('#{}'.format(test_case),end=' ')
    for a in range(10):
        print('{}'.format(sort_list[a]),end=' ')
    print()

# 1 10 1 9 2 8 3 7 4 6 5
# 2 89 8 85 11 67 16 60 28 49 39
# 3 98 3 97 9 88 17 75 18 71 21
