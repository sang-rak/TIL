T = int(input())

for tc in range(1, T+1):
    print('#{}'.format(tc))
    row = int(input())
    for rownum in range(row):
        list = 1
        plist = [list]

        print('1', end=" ")
        for i in range(rownum):
            list = list * (rownum-i) * 1 / (i+1)
            plist.append(int(list))
            print(str(int(list)), end=" ")
        print()


