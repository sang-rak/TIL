'''
13
1 2 1 3 2 4 3 5 3 6 4 7 5 8 5 9 6 10 6 11 7 12 11 13
'''

# import sys
# sys.stdin = open('.', r)

def preorder(node):
    if node != 0:
        print(node, end=" ")
        preorder(tree[node][0])
        preorder(tree[node][1])

def inorder(node):
    if node != 0:
        inorder(tree[node][0])
        print(node, end=" ")
        inorder(tree[node][1])

def postorder(node):
    if node != 0:
        postorder(tree[node][0])
        postorder(tree[node][1])
        print(node, end=" ")
# 정점수
V = int(input())


# 간선수
E = V - 1


# 인접리스트 14 X 3
tree = [[0] * 3 for _ in range(V+1)]
temp = list(map(int, input().split()))

for i in range(E):
    p = temp[i*2]       # 부모
    c = temp[i*2 + 1]   # 자식

    if not tree[p][0]:   # 왼쪽 자식이 없으면
        tree[p][0] = c
    else:                # 왼쪽 자식이 있으면
        tree[p][1] = c

    tree[c][2] = p       # 부모도 저장

preorder(1)
print()
inorder(1)
print()
postorder(1)
print()