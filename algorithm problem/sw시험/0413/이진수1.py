import sys
sys.stdin = open('이진수1.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    N, data = map(str, input().split())

    result = []

    print('#{}'.format(tc), end=' ')

    for i in range(int(N)):
        temp = int(data[i], 16)
        result = '{:04b}'.format(temp)
        print(result, end='')
    print()
