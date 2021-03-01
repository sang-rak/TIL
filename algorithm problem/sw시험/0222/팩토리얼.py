def factorial(n):
    # 종료문 Basic(기본): 멈추는 부분
    if n == 1: 
        return 1
    # 반복문 유도 indective(유도): 자기호풀
    return n * factorial(n-1)

print(factorial(3))