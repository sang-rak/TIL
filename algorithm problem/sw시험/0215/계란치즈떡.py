재료 = ["계란", "치즈", "떡"]

N = 3

for i in range(1<<N):
    ans = ""
    for j in range(N):
        if i & (1<<j):
            ans += 재료[j] + " "
    print( ans, "라면입니다.")

