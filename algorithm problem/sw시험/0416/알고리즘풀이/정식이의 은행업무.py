import sys
sys.stdin = open("정식이의 은행업무.txt","r")

def toDec(x, mode):
    value = 0
    for i in range(len(x)):
        value = value * mode + int(x[i])
    return value

TC = int(input())
for tc in range(1, TC + 1):
    str2 = input()
    str3 = input()

    list2 = []
    list3 = []

    #2진수
    for i in range(len(str2)):
        x2 = list(str2)
        x2[i] = str((int(x2[i]) + 1) % 2)
        list2.append(toDec(x2, 2))

    #3진수
    for i in range(len(str3)):
        for j in [1, 2]:
            x3 = list(str3)
            x3[i] = str((int(x3[i]) + j) % 3)
            list3.append(toDec(x3, 3))

    # 같은값 찾기
    for i in range(len(list2)):
        for j in range(len(list3)):
            if list2[i] == list3[j]:
                ans = list2[i]
                break

    print("#%d" % tc, ans)
