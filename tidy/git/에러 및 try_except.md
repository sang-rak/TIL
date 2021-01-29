1. 에러 종류
   * SytaxError: 문법적오류
   * ZeroDivisionError: 숫자를 0으로 나누려고 할 때
   * NameError: 'abc' is not defined
   * TypeError: 
     * 자료형의 타입이 잘못되었을 때 1+ '3'
     * 매개변수의 갯수와 입력받는 인자의 갯수가 다를 때
   * ValueError: 자료형에 대한 타입은 올바른데 잘못된 값이 입력되는 경우
     * int(3, 5)
   * indexError: list에서 인덱스를 잘못 입력한 경우
   * KeyError: dictionary에서 없는 key로 값 조회를 하는 경우.
   * ModuleNotFoundError: import를 잘못한 경우.
   * ImportError: 모듈은 가져왔는데 그 속에서 클래스나 함수를 찾을 수 없을 때
   * Keyboardinterrupt: ctrl + c 로 종료한 경우
2.  try/ except/ else/ finally

```python
try:
	코드1
	코드2
	코드3
except:
    에러 발생시 실행할 코드
else: # 에러 발생 없이 무사히 코드가 실행이 완료된 경우
    코드5
finally: #에러가 발생하던 말던 try 코드가 실행완료 되면 무조건 실행.
    코드6
```



