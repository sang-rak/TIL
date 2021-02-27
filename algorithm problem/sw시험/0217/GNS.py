
import sys

sys.stdin = open("GNS_test_input.txt", "r")
'''
T = int(input())
# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.

def BubbleSort(x_list):
    for i in range(len(x_list) - 1, 0, -1):  # 4
        for j in range(i):  # 4 3 2 1
            if x_list[j] > x_list[j + 1]:
                x_list[j], x_list[j + 1] = x_list[j + 1], x_list[j]
    return x_list

for test_case in range(1, T + 1):

    N, K = map(str, input().split())
    K = int(K)
    total = list(map(str, input().split()))

    for i in range(K):
        if total[i] == 'ZRO':
            total[i] = 0
        elif total[i] == 'ONE':
            total[i] = 1
        elif total[i] == 'TWO':
            total[i] = 2
        elif total[i] == 'THR':
            total[i] = 3
        elif total[i] == 'FOR':
            total[i] = 4
        elif total[i] == 'FIV':
            total[i] = 5
        elif total[i] == 'SIX':
            total[i] = 6
        elif total[i] == 'SVN':
            total[i] = 7
        elif total[i] == 'EGT':
            total[i] = 8
        else:
            total[i] = 9

    BubbleSort(total)

    for j in range(K):
        if total[j] == 0:
            total[j] = 'ZRO'
        elif total[j] == 1:
            total[j] = 'ONE'
        elif total[j] == 2:
            total[j] = 'TWO'
        elif total[j] == 3:
            total[j] = 'THR'
        elif total[j] == 4:
            total[j] = 'FOR'
        elif total[j] == 5:
            total[j] = 'FIV'
        elif total[j] == 6:
            total[j] = 'SIX'
        elif total[j] == 7:
            total[j] = 'SVN'
        elif total[j] == 8:
            total[j] = 'EGT'
        else:
            total[j] = 'NIN'

    print(N)
    for k in range(K):
        print(total[k],end=' ')
'''
T = int(input())
digit = ["ZRO", "ONE", "TWO", "THR", "FOR", "FIV", "SIX", "SVN", "EGT", "NIN"]
for tc in range(1, T+1):
    temp = input()
    data = list(map(str, input().split()))
    count = [0 for _ in range(10)]
    for i in range(len(data)):
        if data[i] == digit[0]: count[0] += 1
        elif data[i] == digit[1]: count[1] += 1
        elif data[i] == digit[2]: count[2] += 1
        elif data[i] == digit[3]: count[3] += 1
        elif data[i] == digit[4]: count[4] += 1
        elif data[i] == digit[5]: count[5] += 1
        elif data[i] == digit[6]: count[6] += 1
        elif data[i] == digit[7]: count[7] += 1
        elif data[i] == digit[8]: count[8] += 1
        elif data[i] == digit[9]: count[0] += 1

    print("#{}".format(tc), end=" ")
    for i in range(10):
        for j in range(count[i]):
            print(digit[i], end=" ")
    print()