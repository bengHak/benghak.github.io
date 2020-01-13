---
layout: post
title: Flutter 스터디 3주차 요약
subtitle: The Complete 2020 Flutter Development Bootcamp with Dart - created by Angela Yu
tags: [study, Flutter, application, develop, DSC]
comments: true
---

# Section 9: Xylophone - Using Flutter and Dart Packages to Speed Up Development

## Dart Packages

Flutter Package는 open source library code이며
http://pub.dartlang.org/flutter
에서 찾아 사용할 수 있다.

**이번 주차에서 소개한 패키지들**

- cupertino_icons
- generates_words
- shared_preferences 
  (플러터 팀에서 만들어서 성능이 괜찮다고 한다.)
- audioplayers: ^0.13.7

***pubspec.yaml* 파일에 기술해서 사용한다.**



**아래와 같이 패키지 버전을 명시하거나 안할 수 있다.**

```
english_words
english_words: ^3.2.1
```

**설치방법**

```
(안드로이드 스튜디오)
packages get 클릭

(터미널 사용법1)
pub get

(터미널 사용법2)
flutter packages get
```

**사용법**

```
import 'package:<패키지 이름>.dart';
```

## Dart 기초

아래의 기초 개념을 설명했다.

- Function
- Arrow Function



# Section 10:  Quizzler -Modularising & Organising Flutter Code



## Dart 기초

아래의 OOP 개념들을 설명했다.

- IF/ELSE
- Classes and Objects
- Abstraction
- Encapsulation
- Inheritance
- Polymorphism
