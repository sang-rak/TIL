# HTML 



## 1. HTML 기본구조

* 구조
  * head 구조
    * 해당 문서의 정보를 담고 있다.(제목, 문자 인코딩)
    * 외부 로딩 파일 지정도 할 수 있다. (link)
  * body 구조:
    *  브라우저 화면에 나타나는 정보로 실제 내용에 해당한다.
  * DOM tree: 부모관계, 형제관계
  * 요소(element): HTML의 요소는 태그와 내용으로 채워져 있다.
    * 태구 별로 사용하는 속성은 다르다.
    * 시멘틱 태그: 의미론적 요소를 담은 태그
  * 그룹 컨텐츠: p, hr, ol, ul, pre, blockquote, div
  * 텍스트 관련 요소: a, b, i, span, img, em, strong
  * 테이블 관련 요소:tr, td, th, thead, tbody, tfoot, caption, colspan
  * form 태그
    * 입력 받은 데이터와 함께 서버에 요청해주는 태그
    * action: 요청하는 서버의 주소를 설정하는 속성
    * input: 다양한 타입을 가지는 입력 데이터 필드 설정할 수 있음
      * text, checkbox, radio, range, date, ...
      * name (데이터를 담을 이름, 변수명), placeholder, required, disabled, autofocus, ...
      * label tag: 서식의 입력의 이름표, input의 id 값과 연결
  * 속성(attribute): 태그별로 사용할수 있는 속성은 다르다.
  * <a href="https://naver.com"</a>



> ## 시맨틱 태그

* HTML5에서 의미론적 요소를 담은 태그의 등장.
* 대표적인 태그
  * header: 문서 전체나 섹션의 헤더(머릿말 부분)
  * nav: 내비게이션
  * aside: 사이드에 위치한 공간, 메인 컨텐츠와 관련성이 적은 콘텐츠
  * section: 문서의 일반적인 구분, 컨텐츠의 그룹을 표현
  * article: 문서, 페이지, 사이트 안에서 독립적으로 구분되는
  * footer: 문서 전체나 섹션의 푸터(마지막 부분)
* 개발자 및 사용자 뿐만 아니라 검색엔진 등에 의미 있는 정보의 그룹을 태그로 표현
* 단순히 구역을 나누는 것 뿐만 아니라 '의미'를 가지는 태그들을 활용하기 위한 노력
* Non semantic 요소는 div, span 등이 있으며 h1, table 태그들도 시맨틱 태그로 볼 수 있음
* 검색엔진최적화(SEO)를 위해서 메타태그, 시맨틱 태그



> ## 시맨틱 웹

* 다같이 지키자



## 문서 구조화

>## 인라인 / 블록 요소

* 





>## 그룹 컨텐츠

* <a>: 하이퍼링크
* <b>  # 굵게 vs <strong> # 굵게 + 의미강조
* <i> vs <em>: 시멘틱 기울리기
* <span>,  <br> # , <img>
* 기타 등등



> ## table

* <tr>, <td>, <th>
* <thead> #큰머리말, <tbody>#작은머리말, <tfoot> 
* <caption>
* 셀 병합 속성: colspan # 양옆 병합, rowspan # 위아래 병합
* scope 속성
* <col>, <colgroup>



>##  form

* //<form>은 서버에서 처리될 데이터를 제공하는 역할
  * action: 요청할 주소 (입력된 데이터와 함께) 
  * method: http method
  * 인풋은 항상 form태그 내부에 있다.

> ## input

* 다양한 타입을 가지는 입력 데이터 빌드
* <label>: 서식 입력 요소의 캐션
* <input>: 요소의 동작은 type에 따라 달라지므로, 각각의 내용을 숙지 필요
  * MDN web docs
* <input> 공통 속성
  * name # 변수명, placeholder # 흐릿하게 적혀있는 예시문 "이름을 입력하세요"
  * required # 필수입력요소
  * autofocus # 입력창이 깜빡 거리게 하는 요소



> ## 