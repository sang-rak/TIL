# def my_reverse(s):
#     arr = s
#     n = len(arr)
#     for i in range(n//2):
#         arr[i], arr[n-1-i] = arr[n-1-i], arr[i]
#
#     return arr
#
# def itoa(num_str):
#     # 최종 값을 담을 변수
#     value = []
#
#     while True:
#         list_a = num_str % 10
#         value.append(list_a)
#         num_str = num_str // 10
#         if num_str == 0:
#             break
#     value = my_reverse(value)
#     return value

# 교수님 코드
def itoa(num):
    arr = []

    while num:
        y = num % 10
        num //= 10
        arr.append(y)
    arr.reverse()
    return arr
num_str = 1234

num_int = itoa(num_str)
print(num_int, type(num_int))