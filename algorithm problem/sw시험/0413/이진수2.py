import sys
sys.stdin = open('이진수2.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    print('#{}'.format(tc), end=' ')
    data = float(input())

    cnt = 0
    result = []
    while cnt != 12:
        cnt += 1
        temp = data * 2 // 1
        data = data * 2 % 1

        result.append(int(temp))
        if data == 0:
            break

    if data != 0:
        print('overflow')
    else:
        for i in range(len(result)):
            print(result[i], end='')
        print()


