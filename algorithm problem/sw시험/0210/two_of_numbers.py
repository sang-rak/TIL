import sys
sys.stdin = open("two_of_numbers.txt", "r")
T = int(input())
for tc in range(1, T+1):
    # 최대
    An, Bn = list(map(int, input().split())) # An A리스트의 개수 Bn B리스트의 개수
    arr_A = list(map(int, input().split())) # A리스트
    arr_B = list(map(int, input().split()))  # B리스트
    # 개수를 구하고 차이를 빼준다
    if An > Bn:
        ABn = An-Bn
        for i in range(ABn):
            list_ABn = 0
            for j in range(Bn):
                list_ABn += arr_A[i+j] * arr_B[j]
            if i == 0:
                list_max = list_ABn
            else:
                if list_max < list_ABn:
                    list_max = list_ABn
    else:
        ABn = Bn-An
        for i in range(ABn+1):
            list_ABn = 0
            for j in range(An):
                list_ABn += arr_B[i+j] * arr_A[j]
            if i == 0:
                list_max = list_ABn
            else:
                if list_max < list_ABn:
                    list_max = list_ABn
    print(f"#{tc} {list_max}".format(tc, list_max))