import sys
sys.stdin = open("괄호검사.txt",'r')

T = int(input())

for tc in range(1, T+1):
    tmp = str(input())
    tmp_list = []
    for i in range(len(tmp)):

        # 열기
        if tmp[i] == '(' or  tmp[i] == '{':
            tmp_list.append(tmp[i])

        # 닫기
        if tmp[i] == ')':
            if not tmp_list:
                tmp_list.append(')')
            else:
                if tmp_list[-1] == '(':
                    tmp_list.pop()
                else:
                    tmp_list.append(')')

        if tmp[i] == '}':
            if not tmp_list:
                tmp_list.append('}')
            else:
                if tmp_list[-1] == '{':
                    tmp_list.pop()
                else:
                    tmp_list.append('}')
    if not tmp_list:
        print("#{} {}".format(tc, 1))
    else:
        print("#{} {}".format(tc, 0))

