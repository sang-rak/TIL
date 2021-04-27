import sys
sys.stdin = open('단순 2진 암호코드.txt', 'r')

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())


    arr = []
    max = 0
    # 키값 찾기
    for _ in range(N):
        temp = input()

        for i in range(M):

            # print(temp[i], end='')
            if temp[i] != '0':
                if max < i:
                    max = i
                    key = temp[max-55:max+1]

    # 키값 나누기
    key_list = []
    for i in range(8):

        key_list.append(key[i*7:i*7+7])

    # 숫자변환
    key_int = []
    for i in range(len(key_list)):

        if key_list[i] == '0001101':
            key_int.append('0')
        elif key_list[i] == '0011001':
            key_int.append('1')
        elif key_list[i] == '0010011':
            key_int.append('2')
        elif key_list[i] == '0111101':
            key_int.append('3')
        elif key_list[i] == '0100011':
            key_int.append('4')
        elif key_list[i] == '0110001':
            key_int.append('5')
        elif key_list[i] == '0101111':
            key_int.append('6')
        elif key_list[i] == '0111011':
            key_int.append('7')
        elif key_list[i] == '0110111':
            key_int.append('8')
        elif key_list[i] == '0001011':
            key_int.append('9')

    # 검증
    total = (int(key_int[0]) + int(key_int[2]) + int(key_int[4]) + int(key_int[6])) * 3 + int(key_int[1]) + int(key_int[3]) + int(key_int[5]) + int(key_int[7])

    # 검증코드가 맞다면
    result = 0
    if total % 10 == 0:
        for i in range(8):
            result += int(key_int[i])
    else:
        result = 0

    print('#{}'.format(tc), result)