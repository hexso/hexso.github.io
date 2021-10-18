---
layout: post
title: Spring Boot로 게시판 만들기(2)(@OnetoMany, @ManyToOne, DTO, @JsonIgnore, Session)
date: 2021-10-04 01:23:18 +0800
last_modified_at: 2021-10-04 01:23:18 +0800
tags: [BE, Spring Boot]
toc:  true
---



구현한 기능 : 

   1.댓글 조회,추가,삭제, 변경  

2. Service, Controller 구조 정리
3.  Session 구현



### @OnetoMany, @ManyToOne

JPA와 Entity를 이용하여 DB에 접근할 때,  Entity에서 특정 Column이 왜래키(?)로 설정되어 있는 다른 DB의 정보를 가지고 오는데 사용한다.

이 때 하나의 Data에 다른 Table에서 여러개의 DB가 엮여있는 경우 OnetoMany를 사용.

그 반대로 불러오는 Table에서 여러개의 Data가 다른 Table에서 한개의 DB와 엮여있는 경우 ManyToOne을 사용한다.



현재 게시판의 모델

![](/uploads/db/webboardmodel.JPG)





### DTO

Java는 객체로 데이터를 주고 받고 관리를 한다.

이 때 하나의 객체에 여러개의 Data를 가지고 있다고 가정.

그 중 몇개의 data만 전달하고 싶다고 할때 DTO를 이용하여 관리한다.

해당 프로젝트에서 User의  entity는 Password까지 함께가지고 있는데, 이를 response로 전달하면 안되기에 그 사이에 password가 없는 DTO를 이용하여 관리하고 전달할 수 있다.





### @JsonIgnore

Entity와 JPA를 이용하여 Password를 불러왔을때 이를 Client에 response하면 비밀번호까지 함께 전송이 되었다.

DTO를 이용하여 구현하자, 친구가 알려준 방법.

실제 Data가 Json형식으로 변환될때 해당 Data는 변환되지 않는다.

즉 Password에 @JsonIgnore를 해준다면 이 값은 객체일때는 계속 가지고 있으나, 

마지막에 Controller를 통해 return user를 해줄때 password는 전송하지 않는것이다.







### Session

http 프로토콜은 stateless다.  따라서 Login을 했을때 해당 상태를 서버나 클라이언트측에서 저장하지 않는 경우 이 상태를 알 수 없다.

이를 위해 Session,Cookie, Token등을 이용하여 사용자를 관리한다.



**쿠키와 세션 차이**

|                           | **쿠키(Cookie)**                                             | **세션(Session)**                     |
| ------------------------- | ------------------------------------------------------------ | ------------------------------------- |
| **저장 위치**             | 클라이언트(=접속자 PC)                                       | 웹 서버                               |
| **저장 형식**             | text                                                         | Object                                |
| **만료 시점**             | 쿠키 저장시 설정 (브라우저가 종료되도, 만료시점이 지나지 않으면 자동삭제되지 않음) | 브라우저 종료시 삭제 (기간 지정 가능) |
| **사용하는 자원(리소스)** | 클라이언트 리소스                                            | 웹 서버 리소스                        |
| **용량 제한**             | 총 300개 하나의 도메인 당 20개 하나의 쿠키 당 4KB(=4096byte) | 서버가 허용하는 한 용량제한 없음.     |
| **속도**                  | 세션보다 빠름                                                | 쿠키보다 느림                         |
| **보안**                  | 세션보다 안좋음                                              | 쿠키보다 좋음                         |



세션의 경우에는 서버에서 계속 관리를 해야하기 때문에 많은 메모리가 요구가 된다.

따라서 이를 보완하기 위해 나온게 Token

인터넷에 찾아보니 Token은 암호화(RSA, HMAC)를 이용하여 해당 Data를 보호한다.

이 Data자체로 사용자를 인증하는 방식이다.

이 Data는 헤더 + 페이로드 + Signature로 되어 있으며 이를 Server가  조작을 확인하는 방식인것 같다.



#### **Token 구조**

![](/uploads/web/token_model.png)

이를 통해 중간의 공격자가 key를 알지 못하므로 Data자체는 조작할 수 없다.

그러나 Token자체를 탈취하는 경우 해당 동작은 그대로 사용할 수 있는것 같다.

Token 탈취의 가능성이 있으므로 Token을 주기적으로 발급해줌으로 써 위험성을 줄인다.

