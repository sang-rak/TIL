

```python
a = 10 # global
b = 10

def func1(): # enclosed
	a = 30
	def fun2():   #local
		c = 40 
        print(a, b, c)
        
	def fun3():   #local
		c = 40 
        print(a, b, c)
    func2()
    a = 50
    
func1()
```


