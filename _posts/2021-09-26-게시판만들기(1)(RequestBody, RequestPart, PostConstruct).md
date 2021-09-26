---
layout: post
title: Spring Boot로 게시판 만들기(1)(RequestBody, RequestPart, PostConstruct)
date: 2021-09-26 01:23:18 +0800
last_modified_at: 2021-09-26 01:23:18 +0800
tags: [BE, Spring Boot]
toc:  true
---

백엔드를 Spring boot를 통해 공부하던것을 정리하기 위해 작성한다.

처음부터 하나씩 하기에는 막연해서 생각 나는것과 헷갈렸던 것 위주로 작성하기로 했다.



### RequestBody, RequestPart

Amazon S3를 웹서버로 만들어서 이미지 및 동영상을 저장하려고 했다.

이 때 사진이 포함된 HTTP Post 요청을 받으려고 했는데  unsupported media type 에러가 떴다.

찾아보니 @RequestBody로 Post를 요청받고 있었는데, form-data의 경우에는 requestbody로 받을 수가 없었다. 

따라서 @RequestPart("file") MultipartFile file, @RequestPart("body") PostForCreate post

이렇게 두개의 part로 나누어서 요청하고 받도록 하자 무사히 실행되었다.

이때 Post 요청시에도 body에 데이터들을 넣어서 전송해야 한다.





### PostConstruct

AmazonS3를 사용하기 위해 예제를 살펴보던중 @PostConstruct를 보게 되었다.

```java
@PostConstruct
public void setS3Client() {
    AWSCredentials credentials = new BasicAWSCredentials(this.accessKey, 		this.secretKey);

    s3Client = AmazonS3ClientBuilder.standard()
        .withCredentials(new AWSStaticCredentialsProvider(credentials))
        .withRegion(region)
        .build();
}
```

확인해보니 s3Client를 다른 함수에서 사용한다. 이 때 setS3Client를 실행해주어야 s3Client에 연결되게 되는데, 실행해주지 않아도 자동으로 한번 실행하도록 해주는 것이였다. 또한 중복으로 실행되는 경우도 막아주기도 한다.





