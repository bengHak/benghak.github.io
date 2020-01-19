---
layout: post
title: Flutter 스터디 4주차 요약
subtitle: The Complete 2020 Flutter Development Bootcamp with Dart - created by Angela Yu
tags: [study, Flutter, application, develop, DSC]
comments: true
---

# 플러터 테마

플러터 테마에 대해 궁금하다면 cookbook을 보면 좋다고 한다.  
https://flutter.dev/docs/cookbook

## Creating app theme

강의에서는 Material 테마를 사용했다. 테마 색은 'theme' 옵션을 통해 정할 수 있다고 한다.

```dart
class BMICalculator extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ,
      home: InputPage(),
    );
  }
}
```



ThemeData를 커스터마이징 할 수 있다고 한다. 원하는 요소들의 색을 직접 지정할 수도 있다.   
https://api.flutter.dev/flutter/material/ThemeData-class.html  

Colorzila를 통해 웹 브라우저의 색을 추출해서 쓸 수 있다고 한다. 하지만 나는 설치하진 않고, 강의의 컬러코드를 사용했다. https://www.colorzilla.com/  
어플의 Text 테마는 textTheme으로 설정해준다. textTheme의 textStyle로 설정한다.

```dart
class BMICalculator extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData.dark().copyWith(
        primaryColor: Color(0XFF0A0E21),
        scaffoldBackgroundColor: Color(0XFF0A0E21),
        accentColor: Colors.purple,
        textTheme: TextTheme(
          body1: TextStyle(
            color: Colors.white,
          ),
        ),
      ),
      home: InputPage(),
    );
  }
```



Theme 위젯을 통해 특정 위젯의 테마를 커스터마이징 할 수 있다.

```dart
     floatingActionButton: Theme(
        data: ThemeData(
          accentColor: Colors.lightBlue,
        ),
        child: FloatingActionButton(
          child: Icon(Icons.add),
        ),
      ),
```



## Extracting Widjet

복잡해지는 앱에서 반복되는 위젯들을 변수로 사용할 수 있다. 



### Container 컴포넌트 만들기

```dart
body: Container(
  color: Color(0xFF1D1E33),
  margin: EdgeInsets.all(15.0),
  decoration: BoxDecoration(
    borderRadius: BorderRadius.circular(10.0),
  ),
  height: 200.0,
  width: 170.0,
));
```

위와 같이 하면 에러가 난다. 이유는 BoxDecoration 안 쪽에서 color를 정의해줘야 하기 때문이다. 따라서 올바른 정의는 아래와 같다.

```dart
body: Container(
  margin: EdgeInsets.all(15.0),
  decoration: BoxDecoration(
    color: Color(0xFF1D1E33),
    borderRadius: BorderRadius.circular(10.0),
  ),
  height: 200.0,
  width: 170.0,
));
```



### Challenge (고정된 높이, 너비를 사용하지 않고 구현하기)

```dart
import 'package:flutter/material.dart';

class InputPage extends StatefulWidget {
  @override
  _InputPageState createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text('BMI CALCULATOR'),
        ),
        body: Column(
          children: <Widget>[
            Expanded(
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: Container(
                      margin: EdgeInsets.all(15.0),
                      decoration: BoxDecoration(
                        color: Color(0xFF1D1E33),
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      margin: EdgeInsets.all(15.0),
                      decoration: BoxDecoration(
                        color: Color(0xFF1D1E33),
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: Container(
                margin: EdgeInsets.all(15.0),
                decoration: BoxDecoration(
                  color: Color(0xFF1D1E33),
                  borderRadius: BorderRadius.circular(10.0),
                ),
              ),
            ),
            Expanded(
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: Container(
                      margin: EdgeInsets.all(15.0),
                      decoration: BoxDecoration(
                        color: Color(0xFF1D1E33),
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                    ),
                  ),
                  Expanded(
                    child: Container(
                      margin: EdgeInsets.all(15.0),
                      decoration: BoxDecoration(
                        color: Color(0xFF1D1E33),
                        borderRadius: BorderRadius.circular(10.0),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ));
  }
}

```

**Expanded**위젯을 통해서 구현 가능했다.  
하지만 반복되는 코드가 비효율적이어서 새로운 방법을 가르쳐 주실 예정이다.



### ReusableCard class

ReusableCard 라는 statless 위젯을 정의했다.  
위젯의 사용법도 언급하는데 이전 버전의 dart와 dart 2.0에서의 위젯 사용법의 차이를 알려주었다. dart 2.0 에서는 위젯을 사용할 때 **new** 를 사용하지 않아도 된다고 한다.

```dart
// 이전 버전 dart
child: Row(
  children: <Widget>[
    Expanded(
      child: new ReusableCard(),
    ),
      
// dart 2.0
child: Row(
  children: <Widget>[
    Expanded(
      child: ReusableCard(),
    ),
```



**Reusable class 활용한 코드**  
코드 양이 많이 줄어든 모습을 볼 수 있다.

```dart
import 'package:flutter/material.dart';

class InputPage extends StatefulWidget {
  @override
  _InputPageState createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
          title: Text('BMI CALCULATOR'),
        ),
        body: Column(
          children: <Widget>[
            Expanded(
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: ReusableCard(),
                  ),
                  Expanded(
                    child: ReusableCard(),
                  ),
                ],
              ),
            ),
            Expanded(
              child: ReusableCard(),
            ),
            Expanded(
              child: Row(
                children: <Widget>[
                  Expanded(
                    child: ReusableCard(),
                  ),
                  Expanded(
                    child: ReusableCard(),
                  ),
                ],
              ),
            ),
          ],
        ));
  }
}

class ReusableCard extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.all(15.0),
      decoration: BoxDecoration(
        color: Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(10.0),
      ),
    );
  }
}

```



### Key class

위젯의 상태를 나타내지만 필요없으므로 지웠다.  
https://api.flutter.dev/flutter/foundation/Key-class.html  
https://youtu.be/kn0EOS-ZiIc



### Class Constructor

클래스의 생성자를 만들었고, 필수로 초기화해야 하는 변수는 @required 를 통해 선언할 수 있다.

```dart
class ReusableCard extends StatelessWidget {
  
  ReusableCard({@required this.colour});

  Color colour;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: EdgeInsets.all(15.0),
      decoration: BoxDecoration(
        color: colour,
        borderRadius: BorderRadius.circular(10.0),
      ),
    );
  }
}
```



# Immutable변수

```
info: This class (or a class which this class inherits from) is marked as '@immutable', but one or more of its instance fields are not final: ReusableCard.colour (must_be_immutable at [bmi_calculator] lib\input_page.dart:59)
```

ReusableCard 위젯의 Warning 문구에 **Immutable**이란 단어가 나온다. **Immutability**를 통해 변수의 수정을 막는다. 그 방법으로 **final**과 **Const** 두가지가 있다.



**Const**  
Const 변수는 Runtime에서 정의될 수 없다.

```dart
const Color colour;
const int myConst = 2 + 5 * 8;
```



**Final**  
Final은 Runtime에서 정의될 수 있다. 그래서 ReusableCard에서 final을 사용한다.

```dart
final Color colour;
final myFinal = DateTime.now();
```



## Hardcoding을 피하는 법

dart 파일 상단에 아래와 같이 const 변수로 정의해주는 것으로 하드코딩을 피할 수 있다.

```dart
const bottomContainerHeight = 80.0;
const activeCardColour = Color(0xFF1D1E33);
```



# Creating Custom Flutter Widgets

cardChild 변수를 통해 위젯의 child를 새롭게 정의해주었다.

```dart
class ReusableCard extends StatelessWidget {
  ReusableCard({
    @required this.colour,
    this.cardChild,
  });

  final Color colour;
  final Widget cardChild;

  @override
  Widget build(BuildContext context) {
    return Container(
      child: cardChild,
      margin: EdgeInsets.all(15.0),
      decoration: BoxDecoration(
        color: colour,
        borderRadius: BorderRadius.circular(10.0),
      ),
    );
  }
}

```



아래의 패키지를 사용해서 card를 꾸밀 예정이다. 패키지를 추가하고 packages get을 한 뒤에는 **cold restart**가 필요하다.

```yaml
font_awesome_flutter: ^8.4.0
```



## Extracting Widget

cardChild 옵션이 길어서 위젯을 추출하는 것으로 코드 길이를 줄여보려고 한다.

```dart
ReusableCard(
  colour: activeCardColour,
  cardChild: Column(
    mainAxisAlignment: MainAxisAlignment.center,
    children: <Widget>[
      Icon(
        FontAwesomeIcons.mars,
        size: 80.0,
      ),
      SizedBox(
        height: 15.0,
      ),
      Text(
        'MALE',
        style: TextStyle(
          fontSize: 18.0,
          color: Color(0xFF8D8E98),
        ),
      )
    ],
  ),
),
```



**Flutter Outline**에서 ReusableCard의 Column과 그 하위 위젯들을 아래의 방법으로 하나의 위젯으로 추출할 수 있다.

![extract_1](..\img\flutter_review\extract_1.PNG)

![extract_2](..\img\flutter_review\extract_2.PNG)

![extract_3](..\img\flutter_review\extract_3.PNG)

Refactor를 클릭하면 아래와 같이 코드가 짧아지는 것을 볼 수 있다. 

```dart
ReusableCard(
  colour: activeCardColour,
  cardChild: IconContent(),
)
```



그리고 IconContent 클래스가 새롭게 생긴 것을 확인할 수 있다.

```dart
class IconContent extends StatelessWidget {
  const IconContent({
    Key key,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        Icon(
          FontAwesomeIcons.mars,
          size: 80.0,
        ),
        SizedBox(
          height: 15.0,
        ),
        Text(
          'MALE',
          style: TextStyle(
            fontSize: 18.0,
            color: Color(0xFF8D8E98),
          ),
        )
      ],
    );
  }
}

```



## 위젯 나누기

![separte_widget](..\img\flutter_review\separte_widget.PNG)

**reusable_card.dart**

```dart
import 'package:flutter/material.dart';

class ReusableCard extends StatelessWidget {
  ReusableCard({
    @required this.colour,
    this.cardChild,
  });

  final Color colour;
  final Widget cardChild;

  @override
  Widget build(BuildContext context) {
    return Container(
      child: cardChild,
      margin: EdgeInsets.all(15.0),
      decoration: BoxDecoration(
        color: colour,
        borderRadius: BorderRadius.circular(10.0),
      ),
    );
  }
}

```



**icon_content**

```dart
import 'package:flutter/material.dart';

const labelTextStyle = TextStyle(
  fontSize: 18.0,
  color: Color(0xFF8D8E98),
);

class IconContent extends StatelessWidget {
  IconContent({this.icon, this.label});

  final IconData icon;
  final String label;

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: <Widget>[
        Icon(
          icon,
          size: 80.0,
        ),
        SizedBox(
          height: 15.0,
        ),
        Text(
          label,
          style: labelTextStyle,
        )
      ],
    );
  }
}

```



# The GestureDetector Widget

gestureDetector를 활용해서 위젯을 interactive하게 만들 수 있다.

```dart
void updateColour(int gender) {
  // male card pressed
  if (gender == 1) {
    if (maleCardColour == inactiveCardColour) {
      maleCardColour = activeCardColour;
      femaleCardColour = inactiveCardColour;
    } else {
      maleCardColour = inactiveCardColour;
    }
  }
  if (gender == 2) {
    if (femaleCardColour == inactiveCardColour) {
      femaleCardColour = activeCardColour;
      maleCardColour = inactiveCardColour;
    } else {
      femaleCardColour = inactiveCardColour;
    }
  }
}

```

```dart
GestureDetector(
  onTap: () {
    setState(() {
      updateColour(1);
    });
  },
  child: ReusableCard(
    colour: maleCardColour,
    cardChild: IconContent(
      icon: FontAwesomeIcons.mars,
      label: 'MALE',
    ),
  ),
),
```

위의 코드는 updateColour 함수를 ReusableCard를 tap하면 실행하게 한다.



# Enums

```dart
enum EnumName {typeA, typeB, typeC}
print(EnumName.typeA);
```

위와 같은 형식으로 사용한다.



```dart
enum Gender { male, female }
```

위와 같이 성별을 정의해주면 updateColor 함수를 아래와 같이 바꿀 수 있다.

```dart
void updateColour(Gender selectedGender) {
  // male card pressed
  if (selectedGender == Gender.male) {
    if (maleCardColour == inactiveCardColour) {
      maleCardColour = activeCardColour;
      femaleCardColour = inactiveCardColour;
    } else {
      maleCardColour = inactiveCardColour;
    }
  }
  // female card pressed
  if (selectedGender == Gender.female) {
    if (femaleCardColour == inactiveCardColour) {
      femaleCardColour = activeCardColour;
      maleCardColour = inactiveCardColour;
    } else {
      femaleCardColour = inactiveCardColour;
    }
  }
}
```





# Ternary Operator (3항 연산자)

updateColor로 카드의 색을 변화시키는 대신에 3항 연산자로 색을 변화시키는 코드를 작성했다.

```dart
Gender selectedGender;
selectedGender == Gender.male ? activeCardColour : inactiveCardColour
```



# Functions as First Order Objects

함수를 변수로 다룬다. 아래와 같이 사용할 수 있다.

```dart
void main() {
  int result = calculator(3,5,multiply);
  print(result);
}

final Function calculator = (int n1, int n2, Function calculation) {
  return calculation(n1, n2);
}

int add(int n1, int n2) {
  return n1 + n2;
}

int multiply(int n1, int n2) {
  return n1 * n2;
}

```





# Multi-Screen Apps Using Routes and Navigation

각 페이지를 route로 표현할 수 있다. initialRoute를 사용하고 home을 사용하면 안된다.

```dart
import 'package:flutter/material.dart';
import 'screen0.dart';
import 'screen1.dart';
import 'screen2.dart';

void main() => runApp(MyApp());

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      // home: Screen0(),  <== 이렇게 하면 안된다.
      initialRoute: '/',
      routes: {
        '/': (context) => Screen0(),
        '/first': (context) => Screen1(),
        '/second': (context) => Screen2(),
      },
    );
  }
}

```

