import sys
sys.stdin = open("계산기3.txt", 'r')

T = 10
for tc in range(1,T+1):

    N = int(input())
    emp = list(input())

    emp_int = []
    emp_temp = []
    for str in emp:
        if str == '(':
            emp_temp.append(str)

        elif str == ')':
            while emp_temp[-1] != '(':
                temp = emp_temp.pop()
                emp_int.append(temp)
            emp_temp.pop()

        elif str == '+':
            while emp_temp and emp_temp[-1] != '(':
                temp = emp_temp.pop()
                emp_int.append(temp)
            emp_temp.append(str)

        elif str == '*':
            while emp_temp and emp_temp[-1] != '(' and emp_temp[-1] != '+':
                temp = emp_temp.pop()
                emp_int.append(temp)
            emp_temp.append(str)
        else:
            emp_int.append(int(str))
    # print(emp_int)
    while emp_temp:
        temp = emp_temp.pop()
        emp_int.append(temp)

    result = []
    for str in emp_int:

        if str == '*':
            emp2 = result.pop()
            emp1 = result.pop()
            emp3 = emp1 * emp2
            result.append(emp3)

        elif str == '+':
            emp2 = result.pop()
            emp1 = result.pop()
            emp3 = emp1 + emp2
            result.append(emp3)

        else:
            result.append(str)
    print("#{} {}".format(tc, result[0]))


