import sys
sys.stdin = open("피자 굽기.txt", "r")

T = int(input())

for tc in range(1, T+1):
    N, M = map(int, input().split())
    pizza = list(map(int, input().split()))

    # 화덕에 넣기
    firepot = []

    for i in range(N):
        firepot.append((i+1, pizza[i]))

    next_pizza = N
    last_pizza = -1

    while len(firepot) > 1:
        num, cheese = firepot.pop(0)

        cheese //= 2
        last_pizza = num

        # 치즈가 남아있다면
        if cheese:
            firepot.append((num, cheese))
        else:
            if next_pizza < M:
                firepot.append((next_pizza+1, pizza[next_pizza]))
                next_pizza += 1

    print("#{} {}".format(tc, firepot[0][0]))