def BubbleSort(x_list):
    for i in range(len(x_list)-1, 0, -1): # 4
        for j in range(i):  # 4 3 2 1
            if x_list[j] > x_list[j+1]:
                x_list[j], x_list[j+1] = x_list[j+1], x_list[j]
    return x_list

x_list = [55, 7, 78, 12, 42]
BubbleSort(x_list)
print(x_list)

T = int(input())

for tc in range(1, T+1):
    N = int(input())

    number = map(int, input().split())

    p