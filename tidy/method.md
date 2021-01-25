># String 에서 사용하는 method

1. 조회 탐색
   * find : 첫번째 위치 반환. 없으면 -1
   * index: 첫번째 위치 반환. 없으면 오류!
2. 값 변경
   * replace : 바꿀 대상 글자를 새로운 글자로 바꿔서 반환. count 로 횟수 조절가능
   * strip: 공백제거(lstrip, rstrip), 문자를 넣어서 특정 문자만 제거가능
   * split : 문자역을 나누어서 리스트로 반환
   * join : '문자'.join(반복가능한객체) 형식으로 작성가능
3. 문자변형
   * .capitalize() : 앞글자를 대문자로 만들어 반환한다.
   * .title() : 어프스트로피나 공백 이후를 대문자로 만들어 반환한다.
   * .upper(): 모두 대문자로 만들어 반환한다.
   * .lower() : 모두 소문자로 만들어 반환한다.
   * swapcase() : 대<->소 문자로 변경하여 반환한다.
4. 문자열 관련검증
   * isalpha() : 알파벳으로 구성이 되었는지 확인
   * isspace() : 공백인지 확인
   * isupper() : 대문자 인지
   * islower() : 소문자 인지
   * istitle() : title 형식인지? 첫글자마다 대문자인지
   * isdecimal() : 순수 Int로 변환이 가능한 문자열인지
   * isdigit() : 문자열에서 사용된 글자들이 digit인지 확인, 제곱수까지 판별
   * isnumeric() : 분수의 특수 기호도 True로 판정, 특수 로마자도 사용 가능
   * dir('string'): 로 문자열 메소드를 확인 할 수 있다.



># 리스트에서 사용하는 method

1. 값 추가 및 삭제

   .append(): 리스트에 값을 추가할 수 있다.

   .extend() : 리스트를 없에고 각 객체로 넣어준다.	

   * insert(i, x) : 정해진 위치 i에 값을 추가할 때 사용, 범위를 넘으면 마지막에 아이템 추가

   * remove(x): 리스트에서 값이 x인 것을 삭제, 첫번쨰 만나는 x값을 삭제한다. 값이없으면 오류 발생
   * pop(i): 정해진 위치 i 에 있는 값을 삭제하며 그 항목을 반환한다. i 값이 없으면 마지막 항목을 삭제하고 값을 반환한다.
   * clear() : 리스트의 모든항목을 삭제한다.

2. 탐색 및 정력

   * index(x) : x값을 찾아서 해당 index를 반환처음 만나는 x값을 반환, 값이 없으면 에러

   * count(x): 원하는 값의 갯수를 확인가능

   * ```
     sort([reverse=False])
     ```

     :해당 리스트를 정렬한다. 원형 변형하고 None을 반환한다. 리스트함수

     * reverse(True/False)옵션으로 내림차순으로 정렬할 수 있다.
     * sorted(iterable): 해당 리스트를 정렬하고 정렬된 값을 반환한다. 원본을 유지한다. iterable 한 자료도 가능

   * reverse()

   * 정렬없이 값을 뒤집음 None 반환

     * reverse(): 정렬없이 뒤집은 list_reverseiterator object 를 반환.

3. 리스트 복사

   * 복사

     * 얕은복사

       1. slicing 으로 복사

          ```python
          a = [[1,2], [3,4]]
          b = a[:]
          id(a[0])
          ```

          

​		 	

![image-20210125125408018](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20210125125408018.png)

```
import copy
a = [1, 2, [1, 2]]
b = copy.deepcopy(a)

b[2][0] = 3
print(a)
print(b)
```

4. 데이터 분류

   * immutable
     * number, string, bool, range, tuple
   * mutable
     * list, set, dict

5. List Comprehension

   * 간결함
   * pythonic한 코드
   * 가독성이 떨어질 수 있음

   ```python
   [expression for 변수 in iterable]
   
   list(expression for 변수 in iterable)
   ```

6. map

   * ```
     list_1 = [1, 2, 3]
     list_2 = [3, 4, 6]
     
     def addList(a, b)
     	return a + b
     	
     result = list(map(list_1, list_2))
     print(result)
     ```

   

7. filter(fuction, iterable)

   * filter

8. zip()



# Set

* 변경 가능하고, 순서가 없다 . iterable 객체

* 집합의 요소는 유니크 하다 (중복불가)