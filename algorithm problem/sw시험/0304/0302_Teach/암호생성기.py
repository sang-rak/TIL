import sys
sys.stdin = open("암호생성기.txt", "r")

T = 10

for tc in range(1, T+1):

    no = int(input())
    Q = list(map(int, input().split()))

    cnt = 0
    temp = 0

    while True:  # 무조건 한번은 실행
        temp = Q.pop(0)
        temp -= cnt % 5 + 1
        if temp < 0: temp = 0
        Q.append(temp)
        cnt += 1
        if temp == 0:
            break
    print("#{}".format(tc), end=" ")
    for i in range(len(Q)):
        print(Q[i], end=" ")
    print()
        # if 조건식 : break