import sys
sys.stdin = open("flatten_input.txt", "r")
T = 10
for tc in range(1, T + 1):
    N = int(input())
    arr = list(map(int, input().split()))
    for temp in range(N):
        for i in range(len(arr)):
            if i == 0:
                data_max = arr[i]
                data_min = arr[i]
                max_list = i
                min_list = i
            else:
                if data_max >= arr[i]:
                    if data_min > arr[i]:
                        data_min = arr[i]
                        min_list = i
                else:
                    data_max = arr[i]
                    max_list = i
                    if data_min > arr[i]:
                        data_min = arr[i]
                        min_list = i
        if arr[max_list] == arr[min_list]:
            break
        arr[max_list] = data_max - 1
        arr[min_list] = data_min + 1

    for i in range(len(arr)):
        if i == 0:
            data_max = arr[i]
            data_min = arr[i]
            max_list = i
            min_list = i
        else:
            if data_max >= arr[i]:
                if data_min > arr[i]:
                    data_min = arr[i]
                    min_list = i
            else:
                data_max = arr[i]
                max_list = i
                if data_min > arr[i]:
                    data_min = arr[i]
                    min_list = i

    print(f"#{tc} {arr[max_list] - arr[min_list]}".format(tc, arr[max_list], arr[min_list]))
