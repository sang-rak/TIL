import sys
sys.stdin = open("의석이의 세로로 말해요.txt")

T = int(input())

for tc in range(1, T+1):

    # 최대 길이를 담을 값
    max_len = 0
    word = [0] * 5
    for i in range(5):
        word[i] = list(input())

        if len(word[i]) > max_len:
             max_len =len(word[i])
    print('#{}'.format(tc), end=' ')
    # 읽기
    for i in range(max_len):
        for j in range(5):
            if len(word[j]) > i:
                print(word[j][i], end='')
    print()