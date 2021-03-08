N = 20  # 마이쮸의 개수

queue = [(1, 0)]

# (0, 0) [0] : 사람 번호, [1] : 직전까지 받았던 마이쮸의 개수

new_people = 1
last_people = 0

while N > 0:
    num, cnt = queue.pop(0)

    last_people = num # 마지막으로 받으러 온 사람
    cnt += 1 # 저번보다는 하나 더 챙겨가자

    N -= cnt  # num이라는 친구가 cnt개수만큼 가져감
    new_people += 1

    queue.append((num, cnt))  # 맨 뒤로 가서 다시 줄을 섬
    queue.append((new_people, 0))  # 새로운 사람도 줄섬
    print(queue)
print(last_people)
