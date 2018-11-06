**자전거 대여 수요예측 분석**
=============================

본 문서는 Kaggle에 업로드된 '자전거 수요예측 분석'(Bike Sharing Demand)을 마크다운형식으로 편집하여,
Github에 업로드 하기 위하여 작성된 문서입니다.
데이터 출처 \*<https://www.kaggle.com/c/bike-sharing-demand>

------------------------------------------------------------------------

분석 과정 목차
--------------

1.  변수 정의
2.  분석 과정
    -   데이터 구조 확인
    -   사전 가설 수립(Make insight)
    -   EDA / Data preprocessing
    -   Modeling
    -   MSE Checking
3.  결론

------------------------------------------------------------------------

변수 정의
---------

1.  datetime - hourly date + timestamp

2.  season
    -   1 : spring
    -   2 : summer
    -   3 : fall
    -   4 : winter
3.  holiday - whether the day is considered a holiday (휴일)

4.  workingday - whether the day is neither a weekend nor holiday (주말도, 휴일도 아닌 날)

5.  weather
    -   1: Clear, Few clouds, Partly cloudy, Partly cloudy
    -   2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    -   3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    -   4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
6.  temp - temperature in Celsius(섭씨) -&gt; 실제온도.

7.  atemp - "feels like" temperature in Celsius (체감온도)

8.  humidity - relative humidity (습기)

9.  windspeed - wind speed (풍속)

**Train에만 있는 것. (종속변수)**

-   casual - number of non-registered user rentals initiated (비회원의 렌탈수)

-   registered - number of registered user rentals initiated (회원의 렌탈수)

-   count - number of total rentals (토탈렌트)

------------------------------------------------------------------------

분석 과정
---------

### 데이터 구조확인

**Initialize**

``` r
rm(list=ls())
library(ggplot2)  #for plotting
library(corrplot) #for correlation plot
```

    ## corrplot 0.84 loaded

``` r
library(dplyr)    #for %in% function
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library(caret)    #for cross validation
```

    ## Loading required package: lattice

``` r
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

### 가설 사전 수립(Make insights)

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

------------------------------------------------------------------------

### Exproratory Data Analysis (EDA)

**Preparing Total data**

``` r
test[,c('casual', 'registered', 'count')] <- NA
train <- rename(train, 'y' = 'count')
test <- rename(test, 'y' = 'count')
data <- rbind(train, test)
str(data)
```

    ## 'data.frame':    17379 obs. of  12 variables:
    ##  $ datetime  : chr  "2011-01-01 0:00" "2011-01-01 1:00" "2011-01-01 2:00" "2011-01-01 3:00" ...
    ##  $ season    : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ holiday   : int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ workingday: int  0 0 0 0 0 0 0 0 0 0 ...
    ##  $ weather   : int  1 1 1 1 1 2 1 1 1 1 ...
    ##  $ temp      : num  9.84 9.02 9.02 9.84 9.84 ...
    ##  $ atemp     : num  14.4 13.6 13.6 14.4 14.4 ...
    ##  $ humidity  : int  81 80 80 75 75 75 80 86 75 76 ...
    ##  $ windspeed : num  0 0 0 0 0 ...
    ##  $ casual    : int  3 8 5 3 0 0 2 1 1 8 ...
    ##  $ registered: int  13 32 27 10 1 1 0 2 7 6 ...
    ##  $ y         : int  16 40 32 13 1 1 2 3 8 14 ...

**datetime**

``` r
nrow(data)
```

    ## [1] 17379

``` r
length(table(data$datetime))
```

    ## [1] 17379

-&gt; 문자열 데이터이며, 데이터의 분할이 필요함을 확인할 수 있다.

``` r
ggplot(data = data, aes(x = datetime, y = y)) +
  geom_point() +
  labs(title = 'Scatter plot of data',
       subtitle = 'with datetime')
```

    ## Warning: Removed 6493 rows containing missing values (geom_point).

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-5-1.png)

-&gt; 각 월의 20일~마지막일 까지는 test에 포함된 NA값이다.

따라서, 이 빈 구역들의 y값들을 예측하는 것이 목표이다.

**season**

팩터들의 이름을 부여한다.

``` r
str(data$season)
```

    ##  int [1:17379] 1 1 1 1 1 1 1 1 1 1 ...

``` r
data$season[which(data$season == 1)] <- 'spring'
data$season[which(data$season == 2)] <- 'summer'
data$season[which(data$season == 3)] <- 'fall'
data$season[which(data$season == 4)] <- 'winter'
data$season <- factor(data$season,
                         levels = c('spring', 'summer', 'fall', 'winter'))
levels(data$season)
```

    ## [1] "spring" "summer" "fall"   "winter"

계절에 따른 y값의 모양을 상자그림을 통하여 실시하려고 한다.
단 2가지 방법으로 표현해보도록 하자.(이후 표현은 ggplot으로 통일한다.)

**A. Box Plot으로 그리기**

``` r
boxplot(data$y ~ data$season)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-7-1.png)

**B. ggplot을 활용한 그림**

``` r
ggplot(data = train) +
  geom_boxplot(aes(x = factor(season), y = y) ,fill = c('coral', 'coral1', 'coral2', 'coral3')) +
  labs(title = 'Boxplot of Data' ,
       subtitle = 'Grouped by Season' ,
       x = 'Season') +
  scale_x_discrete(labels = levels(data$season))
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-8-1.png)

-&gt; 계절에 따라 y(count)의 값이 크게 변동은 없으나, summer와 fall의 평균이 어느정도 높은것을 확인할 수 있다.

**holiday**

``` r
data$holiday[which(data$holiday == 0)] <- 'holiday'
data$holiday[which(data$holiday == 1)] <- 'non holiday'
data$holiday <- as.factor(data$holiday)
str(data$holiday)
```

    ##  Factor w/ 2 levels "holiday","non holiday": 1 1 1 1 1 1 1 1 1 1 ...

``` r
table(data$holiday)
```

    ## 
    ##     holiday non holiday 
    ##       16879         500

-&gt; Binary data 이며, Holiday와 Non holiday의 비율이 16879 : 500 인것을 확인할 수 있다.

-&gt; count에 영향을 주지 않는다.

``` r
ggplot(data = train, aes(x = holiday, y = y)) +
  geom_boxplot(aes(group = holiday)) +
  aes(fill = holiday) +
  labs(title = 'Boxplot of Data' ,
       subtitle = 'Grouped by holiday')
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-10-1.png)

-&gt; Holiday 유무에 따라 평균의 큰 차이는 없으나 큰 Y 값들이 Holiday에 상대적으로 많은 것을 확인할 수 있다.

**workingday**

``` r
data$workingday[which(data$workingday == 0)] <- 'non workingday'
data$workingday[which(data$workingday == 1)] <- 'workingday'
data$workingday <- as.factor(data$workingday)
table(data$workingday)
```

    ## 
    ## non workingday     workingday 
    ##           5514          11865

-&gt; 1 : 2의 비율로 분포하는 것을 확인할 수 있다.

차이점 확인하기.

``` r
ggplot(data = train, aes(x = workingday, y = y)) +
  geom_boxplot(aes(group = workingday)) +
  aes(fill = workingday) +
  labs(title = 'Boxplot of Data' ,
       subtitle = 'Grouped by working')
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-12-1.png)

-&gt; 평균의 차이가 크지 않은 것을 확인할 수 있다.

**weather**

범주별 데이터 수 확인

``` r
table(data$weather)
```

    ## 
    ##     1     2     3     4 
    ## 11413  4544  1419     3

-&gt; 4(Heavy Rain)인 자료가 단 3개밖에 존재하지 않는다.

차이점 확인

``` r
ggplot(data = data, aes(x = weather, y = y))+
     geom_boxplot(aes(group = weather)) +
     aes(fill = weather) +
     labs(title = 'Boxplot of Data' ,
     subtitle = 'Grouped by weather')
```

    ## Warning: Removed 6493 rows containing non-finite values (stat_boxplot).

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-14-1.png)

-&gt; 범주 4의 치환이 필요함을 확인할 수 있고, 날씨에 따른 y의 변화를 확인할 수 있다.

**Numeric data Exploring**

temp, atemp, humidity, windspeed 모두 수치형 데이터이며, 이중 주목할 것은 temp(온도)와 atemp(체감온도) 일 것이다.
-&gt; 따라서 두 변수는 반드시 강한 상관관계를 가질 것으로 예상된다.

``` r
num_data <- data[, c('temp', 'atemp', 'humidity', 'windspeed')]
cor(num_data)
```

    ##                  temp       atemp    humidity   windspeed
    ## temp       1.00000000  0.98767214 -0.06988139 -0.02312526
    ## atemp      0.98767214  1.00000000 -0.05191770 -0.06233604
    ## humidity  -0.06988139 -0.05191770  1.00000000 -0.29010490
    ## windspeed -0.02312526 -0.06233604 -0.29010490  1.00000000

-&gt; 예상대로 temp와 atemp의 상관관계를 확인할 수 있었다.

**각 연속형 변수의 히스토그램과, 종속변수와의 상관계수 확인**

test data에는 y값이 존재하지 않으므로, train data에서 numeric data를 추출하여 correlation을 계산한다.

``` r
num_train <- train[, c('temp', 'atemp', 'humidity', 'windspeed', 'y')]
cor(num_train)[,5]
```

    ##       temp      atemp   humidity  windspeed          y 
    ##  0.3944536  0.3897844 -0.3173715  0.1013695  1.0000000

-&gt; 각각의 상관계수를 확인할 수 있다.

**Pair plot 그리기**

시각화를 통해서 상관관계를 다시 확인한다.

``` r
corrplot(cor(num_train), method = 'circle', diag = FALSE)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-17-1.png)

-&gt; 상관관계들을 확인할 수 있다.

**Checking Y**

``` r
ggplot(data = data, aes(data$y))+
     geom_histogram()
```

    ## `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.

    ## Warning: Removed 6493 rows containing non-finite values (stat_bin).

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-18-1.png)

-&gt; 왼쪽으로 치우친 것을 확인할 수 있으며, 변수변환의 필요성을 확인할 수 있다.

-&gt; 또한 변수의 성질(0 이상의 정수)에 따라서 Possion regression의 적합을 생각할 수 있다.

------------------------------------------------------------------------

### Data Preprocessing

**datetime**

-   연월일 / 시간 -&gt; 월별로 계절을 나누는게 어느정도 정확하지 않을까?
-   시간대별로 다른 대여수? 버리고 싶지만, 시간대별로 온도가 너무 다르다.

-&gt; 시간대 별로 morning, afternoon, night, dawn 으로 나눠서 파생변수를 만들고 date time을 지우면 어떨까?

**시간 분할해서 plot그려보기.**

``` r
sp <- unlist(strsplit(data$datetime, ":"))
time <- substr(sp[seq(from = 1, to = length(sp), by = 2)], 12, 13)
data$time <- as.integer(time)
ggplot(data = data, aes(x = time, y = y)) +
  geom_point() +
  labs(title = 'Scatter plot of data',
       subtitle = 'with time')
```

    ## Warning: Removed 6493 rows containing missing values (geom_point).

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-19-1.png)

-&gt; 일정한 추세를 보이는 것을 확인할 수 있었다.

**날짜로 시간대 나누기**

0 ~ 6시 dawn, 7 ~ 12시 morning, 1 ~ 6시 afternoon, 7 ~ 12시 night로 범주화시킨다.

``` r
dawnidx <- which(data$time == 0  #data$time %in% paste0(0:5, ":00")
               | data$time == 1
               | data$time == 2
               | data$time == 3
               | data$time == 4
               | data$time == 5)
morningidx <- which(data$time == 6  #data$time %in% paste0(6:11, ":00")
                    | data$time == 7
                    | data$time == 8
                    | data$time == 9
                    | data$time == 10
                    | data$time == 11)
afternoonidx <- which(data$time == 12  #data$time %in% paste0(12:17, ":00")
                      | data$time == 13
                      | data$time == 14
                      | data$time == 15
                      | data$time == 16
                      | data$time == 17)
nightidx <- which(data$time == 18  #data$time %in% paste0(18:23, ":00")
                  | data$time == 19
                  | data$time == 20
                  | data$time == 21
                  | data$time == 22
                  | data$time == 23)

data$daytime <- data$time

data$daytime[dawnidx] <- 'dawn'
data$daytime[morningidx] <- 'morning'
data$daytime[afternoonidx] <- 'afternoon'
data$daytime[nightidx] <- 'night'

data$time <- as.factor(data$time)
data$daytime <- as.factor(data$daytime)
```

**범주화된 변수로 Box plot 그리기**

``` r
ggplot(data = data, aes(x = daytime, y = y)) +
  geom_boxplot(aes(group = daytime)) +
  aes(fill = daytime) +
  labs(title = 'Boxplot of Data' ,
       subtitle = 'Grouped by daytime')
```

    ## Warning: Removed 6493 rows containing non-finite values (stat_boxplot).

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-21-1.png)

-&gt; 각각의 시간에 따른 추세가 중요해보이므로, 범주화 시킨 변수를 삭제한다.

``` r
data <- subset(data, select = -c(daytime))
```

**연도, 월 추출**

``` r
data$year <- substr(data$datetime, 1, 4)
data$year <- as.factor(data$year)
data$month <- substr(data$datetime, 6, 7)
data$month <- as.factor(data$month)
data <- subset(data, select = -c(datetime))
```

-&gt; datetime에서 일자를 사용하지 않는이유 : test와 train의 일자가 모두 다르기 때문.

**season**

EDA과정에서 약간의 영향력을 확인했기 때문에, 따로 처리하지 않도록 한다.

**Holiday**

위에서 확인한 boxplot을 확인해 보았을 때, 변수의 비율도 치우쳐있고 count에 큰 영향을 주는 것 같지 않다. -&gt; 지워버리자. workingday 데이터만 사용

``` r
data <- subset(data, select = -c(holiday))
```

**workingday**

workingday 변수 역시 큰 영향을 끼치지 않아 보이지만, holiday 변수의 삭제로 인해 이 변수는 남겨놓고 분석을 시도해보기로 한다.

**weather**

4번 범주(악천후)에 속하는 데이터가 3개밖에 존재하지 않는 것을 확인했으므로, 이를 다른 범주로 치환해야 한다.

``` r
list(Index = which(data$weather == 4),
     Previous_day = data[which(data$weather == 4)-1,'weather'],
     Level4_day = data[which(data$weather == 4),'weather'],
     Next_day = data[which(data$weather == 4)+1,'weather'])
```

    ## $Index
    ## [1]  5632 11041 14135
    ## 
    ## $Previous_day
    ## [1] 3 3 3
    ## 
    ## $Level4_day
    ## [1] 4 4 4
    ## 
    ## $Next_day
    ## [1] 3 3 3

-&gt; 3개의 데이터 모두 전후일에 날씨변수가 3이었다. 이에 따라서 모든 범주 4를 범주 3으로 치환한다.

``` r
data[which(data$weather == 4), 'weather'] = '3'
data$weather <- as.factor(data$weather)
```

**Numeric data**

Numeric data는 따로 처리하지 않도록 한다.

-&gt; 파생변수를 만들어서 한번에 처리할 수 있다면?

**불쾌지수 변수 만들기**

``` r
discomfort <- with(data, (9/5)*temp-
                     0.55*(1-(humidity/100))*((9/5)*temp-26)+32)
data$discomfort <- discomfort
```

**상관계수 확인하기(불쾌지수 추가)**

``` r
num_data_2 <- data[which(!is.na(data$y)), c('temp', 'atemp', 'humidity', 'windspeed','discomfort', 'y')]
cor(num_data_2)
```

    ##                   temp       atemp    humidity   windspeed  discomfort
    ## temp        1.00000000  0.98494811 -0.06494877 -0.01785201  0.98683890
    ## atemp       0.98494811  1.00000000 -0.04353571 -0.05747300  0.97367141
    ## humidity   -0.06494877 -0.04353571  1.00000000 -0.31860699  0.03457894
    ## windspeed  -0.01785201 -0.05747300 -0.31860699  1.00000000 -0.03726081
    ## discomfort  0.98683890  0.97367141  0.03457894 -0.03726081  1.00000000
    ## y           0.39445364  0.38978444 -0.31737148  0.10136947  0.34551225
    ##                     y
    ## temp        0.3944536
    ## atemp       0.3897844
    ## humidity   -0.3173715
    ## windspeed   0.1013695
    ## discomfort  0.3455122
    ## y           1.0000000

-&gt; temp와 atemp, discomfort와의 multiple correlation을 확인하고, 변수 처리방안을 고민한다.

**windspeed**

Boxplot을 통한 풍속과 날씨의 연관성 확인

``` r
ggplot(data = data, aes(x = weather, y = windspeed)) +
  geom_boxplot(aes(group = weather)) +
  aes(fill = weather) +
  labs(title = 'Boxplot of Data(windspeed)' ,
       subtitle = 'Grouped by weather')
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-29-1.png)

-&gt; 큰 영향을 보이지 않는 것 같아 보인다.

------------------------------------------------------------------------

### Modeling

**Model 1 : Idea 1 을 활용**

``` r
length(train$y)
```

    ## [1] 10886

``` r
sum(train$y == (train$casual + train$registered))
```

    ## [1] 10886

-&gt; y(count) = casual + resistered임을 확인했기 때문에 따로 모형을 적합시켜서 그 합을 구해본다.

**Data partitioning**

``` r
train1 <- data[1 : nrow(train), ]
test1 <- data[-c(1:nrow(train)), ]
```

**환경에 따른 modeling**

``` r
f1_reg <- formula(registered ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+month+discomfort)
f1_cas <- formula(casual ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+month+discomfort)
fit1_reg <- lm(formula = f1_reg, data = train1)
fit1_cas <- lm(formula = f1_cas, data = train1)
```

**가정 확인**

``` r
par(mfrow = c(2,2))
plot(fit1_reg)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-33-1.png)

``` r
plot(fit1_cas)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-33-2.png)

-&gt; 정규성에 대한 가정을 위배한다. 종속변수의 변환이 필요하다는 것을 확인할 수 있다.

**Model 2 : Log-transformation with Model 1**
각각의 종속변수는 0이상의 정수이므로 1을 더한뒤에 log변환을 시도한다.

``` r
f2_reg <- formula(log(registered+1) ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+month+discomfort)
f2_cas <- formula(log(casual+1) ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+month+discomfort)
fit2_reg <- lm(formula = f2_reg, data = train1)
fit2_cas <- lm(formula = f2_cas, data = train1)
```

**가정 확인**

``` r
par(mfrow = c(2,2))
plot(fit1_reg)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-35-1.png)

``` r
plot(fit1_cas)
```

![](Bike_Sharing_Demand_md_files/figure-markdown_github/unnamed-chunk-35-2.png)

-&gt; 변환 전에 비해, 나아진 모습을 보인다.

**Return to Data partition**

Train data를 T\_model과 V\_model data로 분할하여 각 모형을 평가하기로 한다.

caret::createDataPartition을 활용한 data partition

``` r
set.seed(1)

Caret_idx <- createDataPartition(train1$y, p = 0.8, list = FALSE) 
T_model <- train1[Caret_idx, ]
V_model <- train1[-Caret_idx, ]
```

-&gt; set.seed()함수를 사용하여 매 시행마다 같은 데이터셋을 활용할 수 있도록 고정시킨다.

**Model 2 fitting**

``` r
fit2_reg <- lm(formula = f2_reg, data = T_model)
fit2_cas <- lm(formula = f2_cas, data = T_model)
summary(fit2_reg)
```

    ## 
    ## Call:
    ## lm(formula = f2_reg, data = T_model)
    ## 
    ## Residuals:
    ##     Min      1Q  Median      3Q     Max 
    ## -3.2615 -0.2971  0.0339  0.3570  2.1676 
    ## 
    ## Coefficients: (3 not defined because of singularities)
    ##                        Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)           2.3838490  0.1809709  13.173  < 2e-16 ***
    ## seasonsummer          0.6639883  0.0460977  14.404  < 2e-16 ***
    ## seasonfall            0.6738408  0.0454331  14.832  < 2e-16 ***
    ## seasonwinter          0.7212100  0.0326446  22.093  < 2e-16 ***
    ## workingdayworkingday  0.0729495  0.0137337   5.312 1.11e-07 ***
    ## weather2             -0.0410909  0.0157632  -2.607  0.00916 ** 
    ## weather3             -0.5014494  0.0264789 -18.938  < 2e-16 ***
    ## temp                 -0.0057656  0.0082227  -0.701  0.48321    
    ## atemp                 0.0141229  0.0045314   3.117  0.00184 ** 
    ## humidity             -0.0031813  0.0005252  -6.058 1.44e-09 ***
    ## windspeed            -0.0027577  0.0008734  -3.157  0.00160 ** 
    ## time1                -0.6400506  0.0435448 -14.699  < 2e-16 ***
    ## time2                -1.1303038  0.0435883 -25.931  < 2e-16 ***
    ## time3                -1.5876017  0.0438663 -36.192  < 2e-16 ***
    ## time4                -1.8311659  0.0440375 -41.582  < 2e-16 ***
    ## time5                -0.8119424  0.0440862 -18.417  < 2e-16 ***
    ## time6                 0.3914356  0.0440145   8.893  < 2e-16 ***
    ## time7                 1.3016942  0.0439705  29.604  < 2e-16 ***
    ## time8                 2.0014989  0.0432372  46.291  < 2e-16 ***
    ## time9                 1.5762130  0.0436660  36.097  < 2e-16 ***
    ## time10                1.1438816  0.0438802  26.068  < 2e-16 ***
    ## time11                1.2714419  0.0441886  28.773  < 2e-16 ***
    ## time12                1.4830610  0.0448757  33.048  < 2e-16 ***
    ## time13                1.4361734  0.0448322  32.034  < 2e-16 ***
    ## time14                1.3398650  0.0450853  29.718  < 2e-16 ***
    ## time15                1.3998238  0.0451613  30.996  < 2e-16 ***
    ## time16                1.7236399  0.0449310  38.362  < 2e-16 ***
    ## time17                2.1863320  0.0449402  48.650  < 2e-16 ***
    ## time18                2.1002933  0.0445475  47.147  < 2e-16 ***
    ## time19                1.8161482  0.0440340  41.244  < 2e-16 ***
    ## time20                1.5081618  0.0432796  34.847  < 2e-16 ***
    ## time21                1.2578876  0.0435987  28.852  < 2e-16 ***
    ## time22                1.0060773  0.0436741  23.036  < 2e-16 ***
    ## time23                0.6165345  0.0436715  14.118  < 2e-16 ***
    ## year2012              0.5159799  0.0129798  39.752  < 2e-16 ***
    ## month02               0.2020268  0.0315689   6.400 1.64e-10 ***
    ## month03               0.2405865  0.0335848   7.164 8.50e-13 ***
    ## month04              -0.2518621  0.0356096  -7.073 1.64e-12 ***
    ## month05              -0.0131570  0.0325780  -0.404  0.68632    
    ## month06                      NA         NA      NA       NA    
    ## month07              -0.0963706  0.0330926  -2.912  0.00360 ** 
    ## month08              -0.0831549  0.0323243  -2.573  0.01011 *  
    ## month09                      NA         NA      NA       NA    
    ## month10               0.0383520  0.0342617   1.119  0.26301    
    ## month11               0.0222322  0.0316170   0.703  0.48197    
    ## month12                      NA         NA      NA       NA    
    ## discomfort            0.0064446  0.0048757   1.322  0.18627    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 0.593 on 8666 degrees of freedom
    ## Multiple R-squared:  0.8213, Adjusted R-squared:  0.8204 
    ## F-statistic: 926.1 on 43 and 8666 DF,  p-value: < 2.2e-16

-&gt; month와 season계수의 직접적으로 연관되어 있으므로, NA값이 나온다. season변수를 사용하도록 한다.

**Month변수 삭제 후 다시 적합**

Month변수 삭제하기

``` r
data1 <- subset(data, select = -c(month))

train2 <- data1[1 : nrow(train), ]
test2 <- data1[-c(1:nrow(train)), ]
```

수정된 데이터 train2를 통해 다시 모형적합(formula도 수정)

``` r
set.seed(1)

Caret_idx <- createDataPartition(train2$y, p = 0.8, list = FALSE) 
T_model <- train2[Caret_idx, ]
V_model <- train2[-Caret_idx, ]

f2_reg <- formula(log(registered+1) ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+discomfort)
f2_cas <- formula(log(casual+1) ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+discomfort)
fit2_reg <- lm(formula = f2_reg, data = T_model)
fit2_cas <- lm(formula = f2_cas, data = T_model)
```

**모형 2+1 평가**

``` r
pred2_reg <- exp(predict(fit2_reg, newdata = V_model))-1
pred2_cas <- exp(predict(fit2_cas, newdata = V_model))-1
pred2_y <- pred2_reg + pred2_cas

MSE_f2 <- mean((V_model$y-pred2_y)^2)
RMSE_f2 <- sqrt(MSE_f2)
list(MSE = MSE_f2, RMSE = RMSE_f2)
```

    ## $MSE
    ## [1] 8916.861
    ## 
    ## $RMSE
    ## [1] 94.42913

-&gt; MSE : 8916, RMSE : 94를 기록.

**casual과 registered를 분할하지 않고, 바로 적합시킨 모형2(logtransform) 평가**

``` r
f2_y <- formula(log(y+1) ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+discomfort)
fit2_y <- lm(formula = f2_y, data = T_model)
pred2_real_y <- exp(predict(fit2_y, newdata = V_model))-1


MSE_f2_real_y <- mean((V_model$y-pred2_real_y)^2)
RMSE_f2_real_y <- sqrt(MSE_f2_real_y)
list(MSE = MSE_f2_real_y, RMSE = RMSE_f2_real_y)
```

    ## $MSE
    ## [1] 9269.39
    ## 
    ## $RMSE
    ## [1] 96.27767

-&gt; MSE : 9269, RMSE : 96을 기록.
-&gt; Model2의 경우 분할하여 적합시키는게 정확하다.

**Model 3 fitting : Poisson regression**

``` r
f3_reg <- formula(registered ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+discomfort)
f3_cas <- formula(casual ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+discomfort)
f3_y <- formula(y ~ season+workingday+weather+temp+atemp+humidity+windspeed+time+year+discomfort)

fit3_reg <- glm(formula = f3_reg ,
               data = T_model,
               family = poisson)
fit3_cas <- glm(formula = f3_cas ,
               data = T_model,
               family = poisson)
fit3_y <- glm(formula = f3_y ,
               data = T_model,
               family = poisson)
```

**모형 3+1 평가**

``` r
pred3_reg <- predict(fit3_reg, newdata = V_model)
pred3_cas <- predict(fit3_cas, newdata = V_model)
pred3_y <- pred3_reg + pred3_cas

MSE_f3 <- mean((V_model$y-pred3_y)^2)
RMSE_f3 <- sqrt(MSE_f3)
list(MSE = MSE_f3, RMSE = RMSE_f3)
```

    ## $MSE
    ## [1] 65309.28
    ## 
    ## $RMSE
    ## [1] 255.5568

-&gt; MSE : 65309, RMSE : 255를 기록.

**모형 3 평가**

``` r
pred3_real_y <- predict(fit3_y, newdata = V_model)

MSE_f3_real_y <- mean((V_model$y-pred3_real_y)^2)
RMSE_f3_real_y <- sqrt(MSE_f3_real_y)
list(MSE = MSE_f3_real_y, RMSE = RMSE_f3_real_y)
```

    ## $MSE
    ## [1] 66600.8
    ## 
    ## $RMSE
    ## [1] 258.0713

-&gt; MSE : 66600, RMSE : 258을 기록.

**Model 4 fitting : inflated poisson regression **

``` r
#mat_f4 <- subset(T_model, select = c('season','workingday','weather','temp','atemp','humidity',
#                                                        'windspeed','time','year','discomfort'))
#fit4_reg <- zip.mod(T_model$y, mat_f4)
```

제출파일 만들기
===============

pre1 &lt;- predict(Poifit1, newdata = test1) pre2 &lt;- predict(Poifit2, newdata = test1) exppre1 &lt;- exp(pre1) exppre2 &lt;- exp(pre2) final &lt;- exppre1+exppre2 sampleSubmission &lt;- read.csv('sampleSubmission.csv', header = T) sampleSubmission$count &lt;- final write.csv(sampleSubmission, file = 'smapleSubmission5.csv') str(sampleSubmission)

과대산포를 의심할 수 있을 것 같다.
==================================

mean(train1*c**o**u**n**t*)*v**a**r*(*t**r**a**i**n*1count)

제출파일 만들기
===============

pre &lt;- predict(Poifit, newdata = test1) exppre &lt;- exp(pre) sampleSubmission &lt;- read.csv('sampleSubmission.csv', header = T) sampleSubmission$count &lt;- exppre write.csv(sampleSubmission, file = 'smapleSubmission2.csv') str(sampleSubmission)
