import sys
sys.stdin = open("종이붙이기.txt")

T = int(input())
tmp_list = [0 for _ in range(31)]

def paper(cnt):
    if not cnt:
        return 0
    # 20X10은 경우의 수 1 추가
    if cnt == 1:
        return 1

    # 20X20은 경우의 수 3 추가
    if cnt == 2:
        return 3
    if not tmp_list[cnt]:
        tmp_list[cnt] = paper(cnt-1) + 2 * paper(cnt-2)
    return tmp_list[cnt]


for tc in range(1, T+1):
    tmp = int(input())
    cnt = tmp//10
    print("#{} {}".format(tc, paper(cnt)))