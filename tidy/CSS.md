# CSS

* 스타일, 레이아웃 등을 표시하는 방법을 지정하는 언어
* 적용방법
  * 인라인: 해당 태그에 직접 style 속성을 활용 #관리가 힘듬
  * 내부참조: 태그 내에 <style>에 지정, 모든 html에 적용 불가능
  * 외부참조: 외부 CSS파일을 <head>내<link>를 통해 불러오기 #가장 중요

>## CSS 구문

* 선택자
* 선언
* 속성
* 값



> ## 선택자

* HTML 문서에서 특정한 요소를 선택하여 스타일링 하기 위해서는 반드시 선택자라는 개념이 필요하다.

* 기본 선택자
  * 전체 선택자(*), 타입(요소) 선택자
  * 클래스 선택자  ., 아이디 선택자 #, 속성 선택자 [attr]
* 고급 선택자
  * 자손 선택자:띄어쓰기로 구분

    article p { color: red; }

  * 자식 선택자: >로 구분, 바로 아래의 요소

    article > p { color: bule; }

  * 형제 선택자: ~로 구분, 같은 계층(레벨)에 있는 요소

     p ~ section { color: green; } # p와 같은 레벨의 요소 선택

  *  인접 형제 결합자:

    section + p { colot: orange} # 선택자 밑에 p 가 있는 요소 선택

    
* 의사 클래스/요소(pseudo class)
  * 링크, 동적 의사 클래스
  * 구조적 의사 클래스



> ## class 선택자

* 클래스 선택자는 마침표(.)문자로 시작하며 해당 클래스가 적용된 문서의 모든 항목을 선택



> ## id 선택자

* #문자로 시작하며 기본적으로 클래스 선택자와 같은 방식으로 사용
* 그러나 id는 문서당 한번만 사용 할 수 있으며 요소에는 단일 id 값만 적용할 수 있다.



> ## CSS적용 우선순위

* 중요도 ( 사용시 주의 [사용하지않음])
  * !important
* inline style 적용
* id 선택자(파일당 1 개씩) >  class 선택자 > 요소 선택자 >코드 순서

> ## CSS 상속

* CSS는 상속을 통해 부모요소의 속성을 자식에게 상속한다
  
  * text 관련 요소(font, color, text-align), opacity, visibility
  
  
  
* 속성(프로터피)중에는 상속이 되는 것과 되지 않는 것들이 있다.
  
  * box model 관련요소:w, h, p, m, border,), position 관련

> ## CSS (상대) 크기 단위

* px(픽셀)
* % (기준 되는 사이즈에서의 배율(부모)의 배율 결정)
* em (상속 받는 사이즈에서의 비율) # 상속되서 쓰기 어려움
  * 배수단위, 요소에 지정된 사이즈에 상대적인 사이즈를 가짐
* rem (root size의 배율) # 주로 쓴다.
  * 최상위 요소(html)의 기본폰트 16px 사이즈를 기준으로 배수 단위를 가짐.
* Viewport 기준 단위
  * vw, vh, vmin, vmax



* 색상 표현 단위
  * HEX(#000, #000000)
  * RGB/ RGBA
  * 색상명
  * HSL(명도, 채도, 색조)



>## :ballot_box: CSS Box model

* 구성
  * Margin: 테두리 바깥의 외부 여백 배경색을 지정할 수 없다.
  * border: 테두리 영역
  * padding: 테두리 안쪽의 내부 여백 요소에 적용된 배경색
  * content: 글이나 이미지등 요소의 실제 내용



* box-sizing
  * content-box: 기본값, width 의 너비는 content 영역을 기준으로 삼는다.
  * border-box: width 의 너비를 테두리를 기준으로 잡는다.

:

>## :family: 마진 상쇄

* 마진값이 겹치면 더 큰 마진값으로 덮어 씌어 진다.

* 마진상쇄
  * 수직간의 형제 요소에서 주로 발생
  * 큰 사이즈의 마진을 조정해준다.
  * padding을 이용한다.



> ## box-sizing

* 기본적으로 모든 요소의 box-sizing은 순수 content-box
  * Padding을 제외한 순수 contents 영역만을 box로 지정
* 다만 우리가 일반적으로 영역을 볼 때는 border까지의 너비를 100px 보는 것을 원함

:smile_cat: 해결

설정: box-sizing:border-box;



>## CSS Display

* DIsplay
  * block: 가로폭 전체를 차지
    * div, ul, ol, p, hr, form
    * 수평정렬 margin auto 사용
  * inline
    * 컨텐트의 너비 만큼 가로 폭을 차지
    * width, height, margin-top, margin-bottom 지정할 수 없다.
      * line-height로 위아래 간격 조정.
  * inline-block
  * none:화면에서 완전히 없애 버림
    * visiblity:hidden(보여주지만 않을 뿐 그곳에 자리잡고 있음)



>## CSS position

* 문서 상에서 요소를 배치하는 방법을 지정
* static: 디폴드 값(기준 위치)
  * 기본적인 요소의 배치 순서에 따름(좌측 상단))
  * 부모 요소 내에서 배치될때는 부모 요소의 위치를 기준으로 배치된다.
* 아래는 좌표 프로퍼티(top, bottom, left, right)를 사용하여 이동이가능하다(음수 값도 가능)
  * relative: static 위치를 기준으로 이동(상대 위치)
  * absolute: static이 아닌 가장 가까이 있는 부모/조상 요소를 기준으로 이동(절대 위치)
    * 최대 body까지 올라간다.
    * 사용전 기준점을 정해야한다. (relative)
  * fixed: 부모 요소에 관계없이 브라우저를 기준으로 이동(고정위치)
    * 스크롤시에도 항상 같은 곳에 위치
  * sticky: 화면에 보일떄는 위치하고 있다가 스크롤로 해당 내용이 없어질려고 할때 부모요소가 사라질 때까지 fixed가 된다.