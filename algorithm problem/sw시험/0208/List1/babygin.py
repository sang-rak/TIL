num = 456789 #Baby gin 확인할 6자리 수
c = [0] * [2] #6자리 수로부터 각 자리 수를 추출하여 개수를 누적할 리스트

for i in range(6):
    c[num % 10]