import sys

sys.stdin = open("색칠하기_input.txt", "r")
T = int(input())

# 여러개의 테스트 케이스가 주어지므로, 각각을 처리합니다.
for test_case in range(1, T + 1):
    box = [[0] * 10 for i in range(10)]
    N = 0

    n = int(input())
    for i in range(n):
        x1, y1, x2, y2, color = map(int, input().split())

        for j in range(x1, x2+1):
            for k in range(y1, y2+1):
                if color == 1:
                    # 1이 이미있으면 그냥 넘어감
                    if box[j][k] == 1:
                        continue
                    elif box[j][k] == 0:
                        box[j][k] += 1
                    else:
                        box[j][k] = 3
                        N += 1
                else:
                    if box[j][k] == 2:
                        continue
                    elif box[j][k] == 0:
                        box[j][k] += 2
                    else:
                        box[j][k] = 3
                        N += 1
    print('#{} {}'.format(test_case,N))

