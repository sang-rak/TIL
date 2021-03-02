import sys
sys.stdin = open("원재의 메모리 복구하기.txt")

T = int(input())
for tc in range(1, T + 1):
    tmp_list = list(str(input()))
    count = 0
    target = '0'

    for i in range(len(tmp_list)):
        if tmp_list[i] == target:
            pass
        else:
            target = tmp_list[i]
            count += 1
    print("#{} {}".format(tc, count))


    # 1 49
    # 2 1
    # 3 19
    # 4 23
    # 5 15
    # 6 19
    # 7 6
    # 8 27
    # 9 30
    # 10 8