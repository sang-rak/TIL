import sys
sys.stdin = open("암호생성기.txt", "r")

def Enigma(arr):

    while True:
        cnt = 1
        for _ in range(5):
            temp = arr.pop(0)
            temp -= cnt
            if temp <= 0:
                arr.append(0)
                return arr

            arr.append(temp)
            cnt += 1


T = 10

for tc in range(1, T+1):

    t = int(input())

    arr = list(map(int, input().split()))
    Enigma(arr)
    print("#{}".format(t), end=' ')
    for i in range(len(arr)):
        print(arr[i], end=' ')
    print()