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


# 计算预测精度指标，如均方误差(MSE)、平均绝对误差(MAE)、均方根误差(RMSE)
mse <-mean((data$lnscale - data$lnscale_predict)^2)
mae <-mean(abs(data$lnscale - data$lnscale_predict))
rmse<-sqrt(mean((data$lnscale - data$lnscale_predict)^2))

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
dstat <- calculate_dstat(data$lnscale - data$lnscale_predict)
print(paste("方向精度：", dstat))

