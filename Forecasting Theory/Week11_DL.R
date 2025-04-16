# 设定路径
setwd("E:/02_Postgraduate_UCAS/01_Class Tasks/01_Courses/02_预测理论与方法_2024秋_汤玲老师/03_Data")
# 加载包
#install.packages("urca")
#install.packages("dplyr")
library(zoo)
library(tseries)
library(rugarch)
library(readxl)
library(urca)
library(forecast)
library(xts)
library(dplyr)
#carbon_data <- read_excel("C:/path/to/your/carbonprice.xlsx")
#data <- read_excel("北京原煤产量数据.xlsx")
#data <- read.csv("Yulara练习数据 - 副本.csv")
#OR <- data$Diff_Power
data <- read_excel("DL1.xlsx")
data <- log(data)

y=data[,1]
x=data[,2]
y <- ts(y,frequency = 1, start = (2000))
x <- ts(y,frequency = 1, start = (2000))

#数据检验
result_adf_y <- adf.test(y)
print(result_adf_y)
result_adf_x <- adf.test(x)
print(result_adf_x)
result_lb_y<-Box.test(y, type="Ljung-Box")
print(result_lb_y)
result_lb_x<-Box.test(x, type="Ljung-Box")
print(result_lb_x)

#生成滞后变量
data$lag_1 <- lag(data$x,1)
data <- na.exclude(data)
#数据集划分
n <- nrow(data)
train_size <- round(0.8*n) #计算训练集大小，取前80%的数据
train_data <- data[1:train_size, ] #前80%数据作为训练集
#test_data <- carbon_data[(train_size + 1):nrow(carbon_data),] #剩下的数据作为测试集
test_data <- data[(train_size + 1):n, ]

# DL(1) 和上面一样，因为生成的滞后项已经加入x中
#model <- lm(y ~x+lag_1+lag_2,data=train_data)
model <- lm(y ~x+lag_1,data=train_data)
summary(model)
#获取模型信息准则
aic <- AIC(model)
bic <- BIC(model)
print(aic)
print(bic)


# 模型的残差分析(模型的验证)
resi <- residuals(model, standardize = TRUE)
Box.test(resi, lag = 10, type = "Ljung") # 原假设不存在序列自相关


# 预测数据
predicted_values <- predict(model, newdata = test_data)
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
