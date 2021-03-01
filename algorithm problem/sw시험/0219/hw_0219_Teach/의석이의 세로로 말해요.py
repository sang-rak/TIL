import sys
sys.stdin = open("의석이의 세로로 말해요.txt", "r")

T = int(input())

for tc in range(1, T+1):
    word = [0] * 5
    #최대 길이를 담을 값
    max_len = 0

    for i in range(5):
        word[i] = list(input())
        # 입력을 받으면서 최대 길이를 갱신
        if len(word[i]) > max_len:
            max_len = len(word[i])

    # 세로로 읽어보자.
    print("#{}".format(tc), end=" ")
    for i in range(max_len):
        for j in range(5):
            # if len(word[j]) > i:
            #     print(word[j][i], end="")
            try:
                print(word[j][i], end="")
            except:
                continue
    print()