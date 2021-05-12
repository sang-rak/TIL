# vue app

vue create my-first-vue-app

[vue2] babel, eslint 선택

cd my-first-vue-app



vue add router

npm run serve





# 설치된 npm 확인

npm ls -g --depth=0





# lodash 사용

npm i --save lodash



# git 에서 내려받은 파일

requirement txt 같은 package.json에있는 내용을 받기위해 npm i 실행



# axios 추가

- git

  npm i axios

- import 

   import axios from 'axios'



# vuex add

vue add vuex



# Vuex Digest

>  **Vuex를 사용하는 이유?**
>
> 컴포넌트가 많아졌을 경우 데이터의 관리가 어려워짐
>
> 컴포넌트 간 데이터 공유를 편하게 하기 위해서 사용
>
> 

## 1. Vuex 구성요소

- **state**: 데이터
- **getters**: computed와 비슷한 역할
- **mutations**: state를 변경하는 역할 (== state를 조작하면 안된다는 뜻)
- **actions**: state를 비동기적으로 변경하는 역할
  - (참고) mutations를 통해 '간접적으로' state를 변경합니다



## 2. 컴포넌트에서의 활용법

- state, getters => computed 에서 주로 활용

  ```js
  // state
  this.$store.state.키값
  
  // getters
  this.$store.getters.함수명
  ```



- mutations, actions => methods 에서 주로 활용

  ```
  // mutations
  // git에서 commit? => 기록
  // mutations은 state의 변경사항을 기록.
  this.$store.commit('함수명', 매개변수)
  
  // actions
  this.$store.dispatch('함수명', 매개변수)
  ```

  



## 3. helpers

- store에 있는 요소들의 등록을 도와주는 함수

```js
// App.vue

import { mapState } from 'vuex'
import { mapGetters } from 'vuex'
import { mapMutations } from 'vuex'
import { mapActions } from 'vuex'

export default {
    computed: {
        ...mapState(['이름1', '이름2']),
        ...mapGetters(['이름1', '이름2']),
        
    },
    methods: {
        ...mapMutations(['함수명1', '함수명2']),
        ...mapActions(['함수명1', '함수명2']),
    },
}
```

