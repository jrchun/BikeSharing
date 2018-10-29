**자전거 대여 수요예측 분석**
=============================

본 문서는 Kaggle에 업로드된 '자전거 수요예측 분석'(Bike Sharing Demand)을 마크다운형식으로 편집하여,
Github에 업로드 하기 위하여 작성된 문서입니다.
데이터 출처 \*<https://www.kaggle.com/c/bike-sharing-demand>

------------------------------------------------------------------------

분석 과정 목차
--------------

변수 정의
분석 과정
1) 데이터 구조 확인
2) 사전 가설 수립(Make insight)
3) EDA / Data preprocessing 4) Modeling
5) MSE Checking

결론

------------------------------------------------------------------------

변수 정의
---------

1.  datetime - hourly date + timestamp

2.  season

-   1 : spring
-   2 : summer
-   3 : fall
-   4 : winter

1.  holiday - whether the day is considered a holiday (휴일)

2.  workingday - whether the day is neither a weekend nor holiday (주말도, 휴일도 아닌 날)

3.  weather

-   1: Clear, Few clouds, Partly cloudy, Partly cloudy
-   2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
-   3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
-   4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog

1.  temp - temperature in Celsius(섭씨) -&gt; 실제온도.

2.  atemp - "feels like" temperature in Celsius (체감온도)

3.  humidity - relative humidity (습기)

4.  windspeed - wind speed (풍속)

**Train에만 있는 것. (종속변수)**

-   casual - number of non-registered user rentals initiated (비회원의 렌탈수)

-   registered - number of registered user rentals initiated (회원의 렌탈수)

-   count - number of total rentals (토탈렌트)

------------------------------------------------------------------------

분석 과정
---------

### 2.1 데이터 구조확인

**Initialize**

``` r
rm(list=ls())
library(ggplot2)
setwd('C:\\github\\Project\\BikeSharing')
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
```

``` r
colSums(is.na(train))
```

    ##   datetime     season    holiday workingday    weather       temp 
    ##          0          0          0          0          0          0 
    ##      atemp   humidity  windspeed     casual registered      count 
    ##          0          0          0          0          0          0

``` r
colSums(is.na(test))
```

    ##   datetime     season    holiday workingday    weather       temp 
    ##          0          0          0          0          0          0 
    ##      atemp   humidity  windspeed 
    ##          0          0          0

``` r
str(c(train, test))
```

    ## List of 21
    ##  $ datetime  : chr [1:10886] "2011-01-01 0:00" "2011-01-01 1:00" "2011-01-01 2:00" "2011-01-01 3:00" ...
    ##  $ season    : int [1:10886] 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ holiday   : int [1:10886] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ workingday: int [1:10886] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weather   : int [1:10886] 1 1 1 1 1 2 1 1 1 1 ...
    ##  $ temp      : num [1:10886] 9.84 9.02 9.02 9.84 9.84 ...
    ##  $ atemp     : num [1:10886] 14.4 13.6 13.6 14.4 14.4 ...
    ##  $ humidity  : int [1:10886] 81 80 80 75 75 75 80 86 75 76 ...
    ##  $ windspeed : num [1:10886] 0 0 0 0 0 ...
    ##  $ casual    : int [1:10886] 3 8 5 3 0 0 2 1 1 8 ...
    ##  $ registered: int [1:10886] 13 32 27 10 1 1 0 2 7 6 ...
    ##  $ count     : int [1:10886] 16 40 32 13 1 1 2 3 8 14 ...
    ##  $ datetime  : chr [1:6493] "2011-01-20 0:00" "2011-01-20 1:00" "2011-01-20 2:00" "2011-01-20 3:00" ...
    ##  $ season    : int [1:6493] 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ holiday   : int [1:6493] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ workingday: int [1:6493] 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weather   : int [1:6493] 1 1 1 1 1 1 1 1 1 2 ...
    ##  $ temp      : num [1:6493] 10.7 10.7 10.7 10.7 10.7 ...
    ##  $ atemp     : num [1:6493] 11.4 13.6 13.6 12.9 12.9 ...
    ##  $ humidity  : int [1:6493] 56 56 56 56 56 60 60 55 55 52 ...
    ##  $ windspeed : num [1:6493] 26 0 0 11 11 ...

------------------------------------------------------------------------

-&gt; NA값은 존재하지 않는다는 것을 확인할 수 있다.

### 2.2 가설 사전 수립(Make insights)

**Idea 1. train과 test 데이터의 차이점**

train data에 비하여 test데이터는 casual(비회원의 렌트 수)와 registered(회원의 렌트 수)가 존재하지 않는다.
결국 count는 전체의 렌트 수 이므로, 두개의 모델을 적합시켜서 각각에서의 비회원 / 회원의 렌트수를 예측 한 이후에 그걸 더하면 좀 더 정확하지 않을까?
아무래도 casual과 registered는 확실히 경향에 있어서 차이가 있을 것 같다.
casual일때 더 많이 빌릴까? registered일때 더 많이 빌릴까?
이 사람들의 특징에서는 어떠한 차이가 존재할까?

**Idea 2. 기본 가정 확인하기.**

우리가 원하는 종속변수는 특정한 사건(렌트)의 수를 뜻하는 정수이다.
그렇다면 glm을 통해서 poison분포로 적합시킬 수 있지는 않을까?

**Idea 3. 변수간의 관계에 대해서 생각해보기.**

날짜에 관련된 변수가 datetime, holiday, workingday 이렇게 3개나 존재한다. (어쩌면 weather 또한 관련 되어 있을지도)
Weather는 temp, atemp, humidity와 관계가 밀접하지 않을까?
변수간의 Correration 확인해볼수 있을것 같다.

### Data Preprocessing

**datetime**

-   연월일 / 시간 -&gt; 월별로 계절을 나누는게 어느정도 정확하지 않을까?
-   시간대별로 다른 대여수? 버리고 싶지만.. 시간대별로 온도가 너무 다르다.

-&gt; 시간대 별로 morning, afternoon, night, dawn 으로 나눠서 파생변수를 만들고 date time을 지우면 어떨까?

**시간 분할해서 plot그려보기.**

``` r
sp <- unlist(strsplit(train$datetime, ":"))
time <- substr(sp[seq(from = 1, to = length(sp), by = 2)], 12, 13)
time <- as.integer(time)
plot(train$y~time)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-10-1.png)

-&gt; 일정한 추세를 보이는 것을 확인할 수 있었다.

**날짜로 시간대 나누기**

``` r
data$daytime <- data$datetime
data$time <- substr(data$datetime, 12, nchar(data$datetime))
```

**시간나누기**

``` r
sp <- unlist(strsplit(data$datetime, ":"))
time <- substr(sp[seq(from = 1, to = length(sp), by = 2)], 12, 13)
data$daytime <- time
```

**time : 0 ~ 6시 dawn, 7 ~ 12시 morning, 1 ~ 6시 afternoon, 7 ~ 12시 night로 범주화시킨다.**

``` r
dawnidx <- which(data$time == '0:00'
               | data$time == '1:00'
               | data$time == '2:00'
               | data$time == '3:00'
               | data$time == '4:00'
               | data$time == '5:00')
morningidx <- which(data$time == '6:00'
                    | data$time == '7:00'
                    | data$time == '8:00'
                    | data$time == '9:00'
                    | data$time == '10:00'
                    | data$time == '11:00')
afternoonidx <- which(data$time == '12:00'
                      | data$time == '13:00'
                      | data$time == '14:00'
                      | data$time == '15:00'
                      | data$time == '16:00'
                      | data$time == '17:00')
nightidx <- which(data$time == '18:00'
                  | data$time == '19:00'
                  | data$time == '20:00'
                  | data$time == '21:00'
                  | data$time == '22:00'
                  | data$time == '23:00')

data$daytime[dawnidx] <- 'dawn'
data$daytime[morningidx] <- 'morning'
data$daytime[afternoonidx] <- 'afternoon'
data$daytime[nightidx] <- 'night'

data$time <- as.factor(data$time)
data$daytime <- as.factor(data$daytime)
str(data)
```

    ## 'data.frame':    17379 obs. of  14 variables:
    ##  $ datetime  : chr  "2011-01-01 0:00" "2011-01-01 1:00" "2011-01-01 2:00" "2011-01-01 3:00" ...
    ##  $ season    : Factor w/ 4 levels "spring","summer",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ holiday   : Factor w/ 2 levels "holiday","non holiday": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ workingday: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weather   : int  1 1 1 1 1 2 1 1 1 1 ...
    ##  $ temp      : num  9.84 9.02 9.02 9.84 9.84 ...
    ##  $ atemp     : num  14.4 13.6 13.6 14.4 14.4 ...
    ##  $ humidity  : int  81 80 80 75 75 75 80 86 75 76 ...
    ##  $ windspeed : num  0 0 0 0 0 ...
    ##  $ casual    : int  3 8 5 3 0 0 2 1 1 8 ...
    ##  $ registered: int  13 32 27 10 1 1 0 2 7 6 ...
    ##  $ y         : int  16 40 32 13 1 1 2 3 8 14 ...
    ##  $ daytime   : Factor w/ 4 levels "afternoon","dawn",..: 2 2 2 2 2 2 3 3 3 3 ...
    ##  $ time      : Factor w/ 24 levels "0:00","1:00",..: 1 2 13 18 19 20 21 22 23 24 ...

**연도 추출**

``` r
data$datetime <- substr(data$datetime, 1, 4)
data$datetime <- as.factor(data$datetime)
```

**Holiday**

위에서 확인한 boxplot을 확인해 보았을 때, 변수의 비율도 치우쳐있고 count에 큰 영향을 주는 것 같지 않다. -&gt; 지워버리자. workingday 데이터만 사용

``` r
data <- data[,-3]
```

**workingday**

범주형 변수로 변환 후에, 이름을 부여한다.

``` r
data$workingday <- as.factor(data$workingday)
```

**weather**
범주형 변수로 변환

``` r
data$weather <- as.factor(data$weather)
str(data)
```

    ## 'data.frame':    17379 obs. of  13 variables:
    ##  $ datetime  : Factor w/ 2 levels "2011","2012": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ season    : Factor w/ 4 levels "spring","summer",..: 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ workingday: Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
    ##  $ weather   : Factor w/ 4 levels "1","2","3","4": 1 1 1 1 1 2 1 1 1 1 ...
    ##  $ temp      : num  9.84 9.02 9.02 9.84 9.84 ...
    ##  $ atemp     : num  14.4 13.6 13.6 14.4 14.4 ...
    ##  $ humidity  : int  81 80 80 75 75 75 80 86 75 76 ...
    ##  $ windspeed : num  0 0 0 0 0 ...
    ##  $ casual    : int  3 8 5 3 0 0 2 1 1 8 ...
    ##  $ registered: int  13 32 27 10 1 1 0 2 7 6 ...
    ##  $ y         : int  16 40 32 13 1 1 2 3 8 14 ...
    ##  $ daytime   : Factor w/ 4 levels "afternoon","dawn",..: 2 2 2 2 2 2 3 3 3 3 ...
    ##  $ time      : Factor w/ 24 levels "0:00","1:00",..: 1 2 13 18 19 20 21 22 23 24 ...

더 관찰할 만한 변수 없어? 확인해보자.
-------------------------------------

correration of numeric data.
============================

각 범주별로 온도와 count등의 분포를 알고싶다.
=============================================

str(data.all) library(ggplot2)

ggplot(data = data.all, aes(x = season, y = count))+ geom\_point(alpha = 0.1) \#\#봄을 제외하고는 딱히 분포가 보이지 않는다.

ggplot(data = data.all, aes(x = weather, y = count))+ geom\_point(alpha = 0.1) table(data.all$weather) \#weather가 4인 데이터가 단 3개밖에 존재하지 X \#다른범주에 넣어버리자. which(data.all$weather == 4) data.all\[which(data.all$weather == 4),\] data.all\[which(data.all$weather == 4)-1,\] data.all\[which(data.all$weather == 4)+1,\] \#3개의 데이터 모두 전날에 날씨변수가 3이었음. 3으로 통합해도 되겠다. data.all\[which(data.all$weather == 4), 5\] = '3'

str(data.all) table(data.all*w**e**a**t**h**e**r*, *u**s**e**N**A* = ′*a**l**w**a**y**s*′)*d**a**t**a*.*a**l**l*weather &lt;- factor(data.all$weather) \#\#weather level의 4삭제. hist(data.all$windspeed)

par(mfrow = c(2,1)) hist(train*t**e**m**p*)*h**i**s**t*(*t**r**a**i**n*atemp) with(train, plot(count~temp)) with(train, plot(count~atemp)) cor(train*t**e**m**p*, *t**r**a**i**n*atemp) \#\#temp와 atemp의 multicollinearity 발견!

hist(train*t**e**m**p*)*h**i**s**t*(*t**r**a**i**n*temp)

### Modeling

train1 &lt;- data.all\[1:nrow(train), \] test1 &lt;- data.all\[10887:nrow(data.all), \] nrow(test1) head(train1) hist(train1$count) \#변수변환 고려할 것.

환경에 따른 modeling -&gt; casual과 registered를 제외한다.
==========================================================

fit1 &lt;- lm(count ~ datetime+season+workingday+weather+temp+atemp+humidity+windspeed+time, data = train1) summary(fit1) par(mfrow = c(2,2)) plot(fit1) \#정규성가정 개무시. \#MSE pred1 &lt;- predict(fit1, newdata = test1) sum((test1$count-pred1)^2) / ncol(test1) \#RMSE sqrt(sum((test1$count-pred1)^2) / ncol(test1))

변수변환
--------

par(mfrow = c(1,1)) hist(train1*c**o**u**n**t*)*h**i**s**t*(*l**o**g*(*t**r**a**i**n*1count)) hist(sqrt(train1$count)) fit2 &lt;- lm(sqrt(count) ~ datetime+season+workingday+weather+temp+atemp+humidity+windspeed+time, data = train1) summary(fit2) par(mfrow = c(2,2)) plot(fit2)

MSE
===

pred2 &lt;- predict(fit2, newdata = test1) sum((test1$count-pred2)^2) / ncol(test1) \#RMSE sqrt(sum((test1$count-pred2)^2) / ncol(test1)) par(mfrow = c(1,1)) plot(train1$count)

### Idea1 써먹어 보기. count변수는 결국 registered와 casual의 합에 불과하다.

--&gt; 그렇다면 각각의 특성이 다르므로, 2개의 모델을 만들어서 그것의 예측값의 합을 count의 합으로 둔다면?
=========================================================================================================

sum((train1*r**e**g**i**s**t**e**r**e**d* + *t**r**a**i**n*1casual)==train1$count) nrow(train1) \#합이라는 것을 확인.

hist(train1*r**e**g**i**s**t**e**r**e**d*)*h**i**s**t*(*t**r**a**i**n*1casual) \#박스콕스.......쓰고싶다...단점을 확인해봐야겠다. str(train1)

####### 지현이의 조언 : 종속변수가 이산형이니까 회귀분석을 쓰는 건 옳지않은것이 맞고, PoissonRegression을 써야된다.

library(mpath) \#resistered와 casual을 구분하여 추정한 뒤, 합을 구하기.

Poifit1 &lt;- glm(registered ~datetime+season+holiday+workingday+weather+temp+atemp+humidity+windspeed+time+daytime , data = train1, family = poisson)

Poifit2 &lt;- glm(casual ~datetime+season+holiday+workingday+weather+temp+atemp+humidity+windspeed+time+daytime , data = train1, family = poisson)

제출파일 만들기
===============

pre1 &lt;- predict(Poifit1, newdata = test1) pre2 &lt;- predict(Poifit2, newdata = test1) exppre1 &lt;- exp(pre1) exppre2 &lt;- exp(pre2) final &lt;- exppre1+exppre2 sampleSubmission &lt;- read.csv('sampleSubmission.csv', header = T) sampleSubmission$count &lt;- final write.csv(sampleSubmission, file = 'smapleSubmission5.csv') str(sampleSubmission)

과대산포를 의심할 수 있을 것 같다.
==================================

mean(train1*c**o**u**n**t*)*v**a**r*(*t**r**a**i**n*1count)

모델이 합리적인지 확인해보고 싶을땐? k-fold Cross validation을 사용하는 것이 옳은 것 같다.
------------------------------------------------------------------------------------------

Cross Validation
----------------

library(caret)

set.seed(123) train1\_fold &lt;- createFolds(train1$count, k = 10) form &lt;- formula(count ~ datetime + season + holiday + workingday + weather + temp + atemp + humidity + windspeed + time)

cv\_results &lt;- lapply(train1\_fold, function(x) { count\_train &lt;- train1\[-x, \] count\_test &lt;- train1\[x, \] count\_model &lt;- glm(formula = form, data = count\_train, family = poisson) credit\_pred &lt;- exp(predict(count\_model, newdata = count\_test)) credit\_actual &lt;- count\_test$count MSE &lt;- sum((credit\_actual - credit\_pred)^2) / nrow(credit\_actual) return(MSE) })

weather의 변수를 보자
=====================

table(data.all*w**e**a**t**h**e**r*)*w**h**i**c**h*(*d**a**t**a*.*a**l**l*weather == 4) invaliddata &lt;- which(data.all$weather == 4) \#3개의 데이터 data.all &lt;- data.all\[-invaliddata, \] \#삭제 str(cv\_results) mean(unlist(cv\_results))

train1 &lt;- data.all\[1:nrow(train)-1, \] test1 &lt;- data.all\[10886:nrow(data.all), \]

제출파일 만들기
===============

pre &lt;- predict(Poifit, newdata = test1) exppre &lt;- exp(pre) sampleSubmission &lt;- read.csv('sampleSubmission.csv', header = T) sampleSubmission$count &lt;- exppre write.csv(sampleSubmission, file = 'smapleSubmission2.csv') str(sampleSubmission)
