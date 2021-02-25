# 정렬 되지 않은 것
# arr = [4,9,11,23,19,7]
# key = 23
# for i in range(len(arr)):
#     if key == arr[i]:
#         print(i, "에 위치하고 있음")
#         break
# else:
#     print("못찾음.")

# 정렬된 것
arr = [4,7,9,11,19,23]

for i in range(len(arr)):
    if key == arr[i]:
        print(i, "에 위치하고 있음")
        break
    elif key < arr[i]:
        print(i, "번째 까지만 탐색해봄")
        break
else:
    print("못찾음")