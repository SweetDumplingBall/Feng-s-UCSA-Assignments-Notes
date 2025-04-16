# 设定路径
setwd("E:/02_Postgraduate_UCAS/01_Class Tasks/01_Courses/02_预测理论与方法_2024秋_汤玲老师/03_Data")
# 加载包
library(zoo)
library(tseries)
library(rugarch)
library(readxl)
library(urca)
library(forecast)
library(xts)
library(dplyr)
library(car)
library(gvlma)
library(survival)

##**********************ECM模型适用于原序列非平稳，但是同阶单整并能够通过协整检验的情况********************

#导入数据
data <- read_excel("房价和M2.xlsx")
#data <- log(data)
#data <- sapply(data, as.numeric)
y=data[[1]]
x=data[[2]]
#data$lag_1 <- lag(data$x,1)
#data$lag_2 <- lag(data$x,2)
#data$weighted_x <- (1/3)*data$x + (1/3)*data$lag_1 + (1/3)*data$lag_2
#去除含有NA值的观测
#data <- na.exclude(data)
dy <- diff(y,differences=1)
dx <- diff(y,differences=1)
#y <- ts(y,frequency = 1, start = (2000))
#x <- ts(y,frequency = 1, start = (2000))

#平稳性检验
result_adf_y <- adf.test(y,alternative='stationary')
result_adf_x <- adf.test(x,alternative='stationary')
print(result_adf_y)
print(result_adf_x)
#对一阶差分后的数据进行平稳性检验
result_adf_dy <- adf.test(dy,alternative='stationary')
result_adf_dx <- adf.test(dx,alternative='stationary')
print(result_adf_dy)
print(result_adf_dx)
#协整检验
data_matrix <- cbind(y,x)
jj_test <- ca.jo(data_matrix, type ="trace", ecdet ="const",K= 2, spec = "longrun") #看输出结果test一列如果大于10/5/1pct的t值，说明显著
summary(jj_test)


n <- nrow(data)
train_size <- round(0.9*n) #计算训练集大小，取前90%的数据
train_data <- data[1:train_size, ] #前90%数据作为训练集
#test_data <- carbon_data[(train_size + 1):nrow(carbon_data),] #剩下的数据作为测试集
test_data <- data[(train_size + 1):n, ]

#train_data
fit1 <- lm(y ~x,data=train_data)
summary(fit1)
residuals <- resid(fit1)
train_data$residuals_1  <- lag(residuals,1)
train_data$diff_x <- c(NA,diff(train_data$x))
train_data$diff_y <- c(NA,diff(train_data$y))
train_data <- na.exclude(train_data)


fit2 <- lm(diff_y ~ residuals_1+diff_x,data=train_data)
print(fit2)

# 模型的残差分析(模型的验证) 大于0.01才能进行下一步预测，证明能够获取所有数据信息
resi <- residuals(fit2, standardize = TRUE)
Box.test(resi, lag = 10, type = "Ljung") # 原假设不存在序列自相关

# test_data
fit3 <- lm(y ~x,data=test_data)
summary(fit3)
residuals <- resid(fit3)
test_data$residuals_1  <- lag(residuals,1)
test_data$diff_x <- c(NA,diff(test_data$x))
test_data$diff_y <- c(NA,diff(test_data$y))
test_data <- na.exclude(test_data)
# 预测数据
predicted_values <- predict(fit2, newdata = test_data)
print(predicted_values)

# 计算预测精度指标，如均方误差(MSE)、平均绝对误差(MAE)、均方根误差(RMSE)
mse <-mean((predicted_values - test_data$y)^2)
mae <-mean(abs(predicted_values - test_data$y))
rmse<-sqrt(mean((predicted_values - test_data$y)^2))

#输出预测精度指标
print(paste("均方误差(MSE):", mse))
print(paste("平均绝对误差(MAE):",mae))
print(paste("均方根误差(RMSE):",rmse))

# Dstat方向精度
calculate_dstat <- function(actual, predicted) {
  N <- length(actual)
  a <- rep(0, N - 1)
  
  for (t in 1:(N - 1)) {
    if ((actual[t + 1] - actual[t] > 0 && predicted[t + 1] - predicted[t] > 0) ||
        (actual[t + 1] - actual[t] < 0 && predicted[t + 1] - predicted[t] < 0)) {
      a[t] <- 1
    }
  }
  dstat <- sum(a) / (N - 1)
  return(dstat)
}
dstat <- calculate_dstat(test_data$y, predicted_values)
print(paste("方向精度：", dstat))


#以下无关
model <- lm(y ~x+lag_1+lag_2,data=train_data)
summary(model)
vif(model)

model <- lm(y ~weighted_x,data=train_data)
summary(model)
vif(model)

str(lung)
lung <- na.omit(lung)
#查看数据集的维度
dim(lung)
#查看数据集的变量名
names(lung)
#对数据集进行简单的统计分析
summary(lung)
#相关系数
xx <- cor(data)
print(xx)
#多重共线性回归
fit1 <- lm(formula=wt.loss ~ age+sex+ph.ecog+ph.karno+pat.karno+meal.cal, data = lung)
summary(fit1)
vif(fit1)

result_lb_y<-Box.test(y, type="Ljung-Box")
print(result_lb_y)
result_lb_x<-Box.test(x, type="Ljung-Box")
print(result_lb_x)