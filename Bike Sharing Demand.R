#Initialization
rm(list=ls())


##Loading data
setwd('C:\\Users\\jrchu\\Desktop\\Study\\Project\\Bike sharing demand[Kaggle]')
train <- read.csv('train.csv', stringsAsFactors = F)
test <- read.csv('test.csv', stringsAsFactors = F)
colSums(is.na(train))
colSums(is.na(test))
str(c(train, test))
#NA값은 존재하지 않는다.

### Make insight!

##Idea 1.
#train data에 비하여 test데이터는 casual(비회원의 렌트 수)와 registered(회원의 렌트 수)가 존재하지 않는다.
#결국 count는 전체의 렌트 수 이므로, 두개의 모델을 적합시켜서 각각에서의 비회원 / 회원의 렌트수를 예측 한 이후에 그걸 더하면
#좀 더 정확하지 않을까? 아무래도 casual과 registered는 확실히 경향에 있어서 차이가 있을 것 같다.
#casual일때 더 많이 빌릴까? registered일때 더 많이 빌릴까? 이 사람들의 특징에서는 어떠한 차이가 존재할까?

##Idea 2.
#가정을 확인해보기.
#우리가 원하는 종속변수는 특정한 사건(렌트)의 수를 뜻하는 정수이다.
#그렇다면!! glm을 통해서 poison분포로 적합시킬 수 있지는 않을까?

##Idea 3.
#변수간의 관계에 대해서 생각해보기.
#날짜에 관련된 변수가 datetime, holiday, workingday 이렇게 3개나 존재한다. (어쩌면 weather 또한 관련 되어 있을지도)
#Weather는 temp, atemp, humidity와 관계가 밀접하지 않을까?
#변수간의 Correration 확인해볼수 있을것 같다.



###Exproratory Data Analysis (EDA)
test[,c('casual', 'registered', 'count')] <- NA

colnames(train)[12] <- 'y'
colnames(test)[12] <- 'y'
data <- rbind(train, test)

str(data)
###Plotting function by ggplot
library(ggplot2)
plotting_C <- function(D, X) {
  p1<- ggplot(data = D, aes(x = X)) +
    geom_bar()+
    aes(fill = X) +
    labs(title = 'Histogram of Data' ,
         subtitle = 'by Categorical Variable',
         x = 'Variable')
  return(p1)
}


##datetime

#모두 다른값을 보임을 알 수 있음.

##season
str(data$season)

data$season[which(data$season == 1)] <- 'spring'
data$season[which(data$season == 2)] <- 'summer'
data$season[which(data$season == 3)] <- 'fall'
data$season[which(data$season == 4)] <- 'winter'

data$season <- as.factor(data$season)
levels(data$season)
data$season <- factor(data$season, levels(data$season)[c(2, 3, 1, 4)])

str(data$season)
levels(data$season)
boxplot(data$y ~ data$season)

# ggplot(data = train, aes(x = season, y = y)) +
#   geom_boxplot(aes(group = season)) +
#   aes(fill = season) +
#   labs(title = 'Boxplot of Data' ,
#        subtitle = 'Grouped by Season') +
#   scale_x_discrete(labels = c('one','two','three', 'four'))
#X값이 이름이 아닌 팩터값(1, 2, 3, 4)로 나오는 문제 해결해야함.
# --> Boxplot을 그릴때, group으로 묶지 않고,aes 내에서 factor로 묶은 뒤에 scale_x_discrete로 명명해준다.



ggplot(data = train) +
  geom_boxplot(aes(x = factor(season), y = y) ,fill = c('coral', 'coral1', 'coral2', 'coral3')) +
  labs(title = 'Boxplot of Data' ,
       subtitle = 'Grouped by Season' ,
       x = 'Season') +
  scale_x_discrete(labels = levels(data$season))

#계절에 따라 y(count)의 값이 크게 변동은 없으나, summer와 fall의 평균이 어느정도 높은것을 확인할 수 있다.

##holiday
str(data$holiday)
unique(data$holiday)
table(data$holiday)
#binary data
data$holiday[which(data$holiday == 0)] <- 'holiday'
data$holiday[which(data$holiday == 1)] <- 'non holiday'
data$holiday <- as.factor(data$holiday)

boxplot(data$y ~ data$holiday)

ggplot(data = train, aes(x = holiday, y = y)) +
  geom_boxplot(aes(group = holiday)) +
  aes(fill = holiday) +
  labs(title = 'Boxplot of Data' ,
       subtitle = 'Grouped by holiday')
#holiday 유무에 따라 평균의 큰 차이는 없으나 큰 값들이
#상대적으로 많은 것을 확인할 수 있다.

###Data Preprocessing!
##변수 별 성질과 특성을 고려하여, 정확한 type으로 변환하기.


#datetime : - 연월일 / 시간 -> 월별로 계절을 나누는게 어느정도 정확하지 않을까?

#           - 시간대별로 다른 대여수? 버리고 싶지만.. 시간대별로 온도가 너무 다르다.

#시간대 별로 morning, afternoon, night, dawn 으로 나눠서 파생변수를 만들고 date time을 지우면 어떨까?
#시간 분할해서 plot그려보기.
sp <- unlist(strsplit(train$datetime, ":"))
time <- substr(sp[seq(from = 1, to = length(sp), by = 2)], 12, 13)
plot(train$count~time)
#일정한 추세를 보이는 것을 확인할 수 있었다.

#각 월별로 19일까지는 trainm, 20일부터는 test데이터이다. 연도는 살리자!
#time : 0 ~ 6시 dawn, 7 ~ 12시 morning, 1 ~ 6시 afternoon, 7 ~ 12시 night
test$casual <- NA
test$registered <- NA
test$count <- NA
colnames(test) <- colnames(train)
data.all <- rbind(train, test)
str(data.all)

substr(data.all$datetime, 6, 7) #일정 문자를 추출한다.(월) -> season으로 대체할 수 있을 것 같다.
substr(data.all$datetime, 9, 10) #일정 문자를 추출한다.(날짜)
substr(data.all$datetime, 12, nchar('data.all$datetime')) #일정 문자를 추출한다.(시간)

#날짜로 시간대 나누기.
data.all$time <- substr(data.all$datetime, 12, nchar('data.all$datetime'))
data.all$daytime <- data.all$time

#시간나누기
sp <- unlist(strsplit(data.all$datetime, ":"))
time <- substr(sp[seq(from = 1, to = length(sp), by = 2)], 12, 13)
data.all$daytime <- time

dawnidx <- which(data.all$time == '0:00'
               | data.all$time == '1:00'
               | data.all$time == '2:00'
               | data.all$time == '3:00'
               | data.all$time == '4:00'
               | data.all$time == '5:00')
morningidx <- which(data.all$time == '6:00'
                    | data.all$time == '7:00'
                    | data.all$time == '8:00'
                    | data.all$time == '9:00'
                    | data.all$time == '10:00'
                    | data.all$time == '11:00')
afternoonidx <- which(data.all$time == '12:00'
                      | data.all$time == '13:00'
                      | data.all$time == '14:00'
                      | data.all$time == '15:00'
                      | data.all$time == '16:00'
                      | data.all$time == '17:00')
nightidx <- which(data.all$time == '18:00'
                  | data.all$time == '19:00'
                  | data.all$time == '20:00'
                  | data.all$time == '21:00'
                  | data.all$time == '22:00'
                  | data.all$time == '23:00')

data.all$daytime[dawnidx] <- 'dawn'
data.all$daytime[morningidx] <- 'morning'
data.all$daytime[afternoonidx] <- 'afternoon'
data.all$daytime[nightidx] <- 'night'

data.all$time <- as.factor(data.all$time)
data.all$daytime <- as.factor(data.all$daytime)
str(data.all)

#연도만 남기기.
data.all$datetime <- substr(data.all$datetime, 1, 4) #일정 문자를 추출한다.(연도)
data.all$datetime <- as.factor(data.all$datetime)

str(data.all)
#           - 연도별 렌트의 증가추세? -> 새로운 과제로 발견 가능.


#Season data 이름 부여하기.
data.all$season[data.all$season == 1] <- 'Spring'
data.all$season[data.all$season == 2] <- 'Summer'
data.all$season[data.all$season == 3] <- 'Autumn'
data.all$season[data.all$season == 4] <- 'Winter'

data.all$season <- as.factor(data.all$season)

# #holiday -> 20일(500개)의 데이터밖에 없는 휴일데이터 꼭 필요한가?
# #지워버리자. workingday 데이터로만 써도 될것 같다.
# colnames(data.all)
# sum(data.all$holiday)
# sum(train$holiday)
# sum(test$holiday)
# data.all1 <- data.all[, -3]
str(data.all)
data.all$holiday <- as.factor(data.all$holiday)

#Holiday와 Count는 상관관계가 있을 수도 있는데, 막 지우는건 아닌 것 같다.
  ##범주형변수와 연속형범주의 상관관계는 어떻게 알아낼 수 있지?

#workingday -> factor로 변환
data.all$workingday <- as.factor(data.all$workingday)

#weather -> factor로 변환
data.all$weather <- as.factor(data.all$weather)

str(data.all)

##더 관찰할 만한 변수 없어? 확인해보자.
#correration of numeric data.

#각 범주별로 온도와 count등의 분포를 알고싶다.
str(data.all)
library(ggplot2)

ggplot(data = data.all, aes(x = season, y = count))+
  geom_point(alpha = 0.1)
##봄을 제외하고는 딱히 분포가 보이지 않는다.

ggplot(data = data.all, aes(x = weather, y = count))+
  geom_point(alpha = 0.1)
table(data.all$weather)
#weather가 4인 데이터가 단 3개밖에 존재하지 X
#다른범주에 넣어버리자.
which(data.all$weather == 4)
data.all[which(data.all$weather == 4),]
data.all[which(data.all$weather == 4)-1,]
data.all[which(data.all$weather == 4)+1,]
#3개의 데이터 모두 전날에 날씨변수가 3이었음. 3으로 통합해도 되겠다.
data.all[which(data.all$weather == 4), 5] = '3'

str(data.all)
table(data.all$weather, useNA = 'always')
data.all$weather <- factor(data.all$weather)
##weather level의 4삭제.
hist(data.all$windspeed)

par(mfrow = c(2,1))
hist(train$temp)
hist(train$atemp)
with(train, plot(count~temp))
with(train, plot(count~atemp))
cor(train$temp, train$atemp)
##temp와 atemp의 multicollinearity 발견!

hist(train$temp)
hist(train$temp)






###Modeling
train1 <- data.all[1:nrow(train), ]
test1 <- data.all[10887:nrow(data.all), ]
nrow(test1)
head(train1)
hist(train1$count) #변수변환 고려할 것.

#환경에 따른 modeling -> casual과 registered를 제외한다.
fit1 <- lm(count ~ datetime+season+workingday+weather+temp+atemp+humidity+windspeed+time, data = train1)
summary(fit1)
par(mfrow = c(2,2))
plot(fit1)
#정규성가정 개무시.
#MSE
pred1 <- predict(fit1, newdata = test1)
sum((test1$count-pred1)^2) / ncol(test1)
#RMSE
sqrt(sum((test1$count-pred1)^2) / ncol(test1))

##변수변환
par(mfrow = c(1,1))
hist(train1$count)
hist(log(train1$count))
hist(sqrt(train1$count))
fit2 <- lm(sqrt(count) ~ datetime+season+workingday+weather+temp+atemp+humidity+windspeed+time, data = train1)
summary(fit2)
par(mfrow = c(2,2))
plot(fit2)

#MSE
pred2 <- predict(fit2, newdata = test1)
sum((test1$count-pred2)^2) / ncol(test1)
#RMSE
sqrt(sum((test1$count-pred2)^2) / ncol(test1))
par(mfrow = c(1,1))
plot(train1$count)


###Idea1 써먹어 보기. count변수는 결국 registered와 casual의 합에 불과하다.
# --> 그렇다면 각각의 특성이 다르므로, 2개의 모델을 만들어서 그것의 예측값의 합을 count의 합으로 둔다면?
sum((train1$registered+train1$casual)==train1$count)
nrow(train1)
#합이라는 것을 확인.

hist(train1$registered)
hist(train1$casual)
#박스콕스.......쓰고싶다...단점을 확인해봐야겠다.
str(train1)

#######지현이의 조언 : 종속변수가 이산형이니까 회귀분석을 쓰는 건 옳지않은것이 맞고, PoissonRegression을 써야된다.
library(mpath)
#resistered와 casual을 구분하여 추정한 뒤, 합을 구하기.

Poifit1 <- glm(registered ~datetime+season+holiday+workingday+weather+temp+atemp+humidity+windspeed+time+daytime ,
              data = train1,
              family = poisson)

Poifit2 <- glm(casual ~datetime+season+holiday+workingday+weather+temp+atemp+humidity+windspeed+time+daytime ,
               data = train1,
               family = poisson)



#제출파일 만들기
pre1 <- predict(Poifit1, newdata = test1)
pre2 <- predict(Poifit2, newdata = test1)
exppre1 <- exp(pre1)
exppre2 <- exp(pre2)
final <- exppre1+exppre2
sampleSubmission <- read.csv('sampleSubmission.csv', header = T)
sampleSubmission$count <- final
write.csv(sampleSubmission, file = 'smapleSubmission5.csv')
str(sampleSubmission)

#과대산포를 의심할 수 있을 것 같다.

mean(train1$count)
var(train1$count)


##모델이 합리적인지 확인해보고 싶을땐? k-fold Cross validation을 사용하는 것이 옳은 것 같다.
##
##Cross Validation
library(caret)

set.seed(123)
train1_fold <- createFolds(train1$count, k = 10)
form <- formula(count ~ datetime + season + holiday + workingday + weather + 
                  temp + atemp + humidity + windspeed + time)

cv_results <- lapply(train1_fold, function(x) {
  count_train <- train1[-x, ]
  count_test <- train1[x, ]
  count_model <- glm(formula = form,
                     data = count_train,
                     family = poisson)
  credit_pred <- exp(predict(count_model, newdata = count_test))
  credit_actual <- count_test$count
  MSE <- sum((credit_actual - credit_pred)^2) / nrow(credit_actual)
  return(MSE)
})

#weather의 변수를 보자
table(data.all$weather)
which(data.all$weather == 4)
invaliddata <- which(data.all$weather == 4)
#3개의 데이터
data.all <- data.all[-invaliddata, ]
#삭제
str(cv_results)
mean(unlist(cv_results))

train1 <- data.all[1:nrow(train)-1, ]
test1 <- data.all[10886:nrow(data.all), ]



#제출파일 만들기
pre <- predict(Poifit, newdata = test1)
exppre <- exp(pre)
sampleSubmission <- read.csv('sampleSubmission.csv', header = T)
sampleSubmission$count <- exppre
write.csv(sampleSubmission, file = 'smapleSubmission2.csv')
str(sampleSubmission)

