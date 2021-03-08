#  front, rear 이용
SIZE = 100
Q = [0]*SIZE
front, rear = -1, -1
def isFull():
    return len(Q)-1 == rear

def isEmpty():
    return front == rear

def enQueue(item):
    global rear
    if isFull(): print("Queue Full")
    else:
        rear += 1
        Q[rear] = item
def deQueue():
    global front
    if front != rear:
        front += 1
        return Q[front]

def Qpeek():

    global front, rear
    if front != rear:
    
        return Q[front+1]
enQueue(1)
enQueue(2)
enQueue(3)
print(Qpeek())
print(deQueue())
print(deQueue())
print(deQueue())