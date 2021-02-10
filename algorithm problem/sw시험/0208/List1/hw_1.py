
import sys
# 정렬 코드
def BubbleSort(x_list):
    for i in range(len(x_list)-1, 0, -1): # 4
        for j in range(i):  # 4 3 2 1
            if x_list[j] > x_list[j+1]:
                x_list[j], x_list[j+1] = x_list[j+1], x_list[j]
    return x_list

sys.stdin = open('input.txt')
for t in range(10):
    N = int(input())
    df = list(map(int, input().split()))
    # 조경권 확보



# 조경권 확보 코드
    count = 0
    for i in range(2, len(df)-2):
        # 2칸이상 공백이 있는가
        if (df[i] > df[i-1]) & (df[i] > df[i-2]) & (df[i] > df[i+1]) & (df[i] > df[i+2]):

            df_ab = [0, 0, 0, 0]
            df_ab[0] += df[i] - df[i - 2]
            df_ab[1] += df[i] - df[i - 1]
            df_ab[2] += df[i] - df[i + 1]
            df_ab[3] += df[i] - df[i + 2]
            BubbleSort(df_ab)
            count += df_ab[0]

    print(count)


