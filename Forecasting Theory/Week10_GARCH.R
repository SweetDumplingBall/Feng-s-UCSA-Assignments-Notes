# 设定路径
setwd("E:/02_Postgraduate_UCAS/01_Class Tasks/01_Courses/02_预测理论与方法_2024秋_汤玲老师/03_Data")
# 加载包
#install.packages("urca")
#install.packages("xts")
library(zoo)
library(tseries)
library(rugarch)
library(readxl)
library(urca)
library(forecast)
library(xts)
#carbon_data <- read_excel("C:/path/to/your/carbonprice.xlsx")
data <- read_excel("北京原煤产量数据.xlsx")
#data <- read.csv("差分_Yulara练习数据.csv")
OR <- data$d_production

# 创建起始和结束时间的POSIXct对象
#start_date <- as.POSIXct("2023-11-01 00:00:00", tz = "UTC")
#end_date <- as.POSIXct("2024-10-31 23:55:00", tz = "UTC")
#time_index <- seq(from = start_date, to = end_date, by = "5 mins")
#print(time_index)

#ts_data <- xts(OR, order.by = time_index)

#1.单位根检验/纯随机性检验
#对数据平稳性及纯随机性进行检验
#DF检验
ur.df(OR)
summary(ur.df(OR))
#ADF检验
result_adf <- adf.test(OR)
print(result_adf)
#pp检验
pp_test <- ur.pp(OR, type="Z-tau", model="constant", lags="short")
summary(pp_test)
#纯随机检验
result_lb<-Box.test(OR, type="Ljung-Box")
print(result_lb)

#画自相关和偏自相关的图
#install.packages("ggplot2")
#install.packages("ggfortify")
library(ggplot2)
library(ggfortify)
# 绘制自相关图（ACF）
acf(train_data, main="自相关图（ACF）") + theme_minimal()
# 绘制偏自相关图（PACF）
pacf(train_data, main="偏自相关图（PACF）") + theme_minimal()

#autoplot(acf(train_data, lag.max=20)) + ggtitle("自相关图（ACF）")
#autoplot(pacf(train_data, lag.max=20)) + ggtitle("偏自相关图（PACF）")

#区分训练集和测试集
n <- length(OR)
train_size <- round(0.8*n) #计算训练集大小，取前80%的数据
train_data <- OR[1:train_size] #前80%数据作为训练集
#test_data <- carbon_data[(train_size + 1):nrow(carbon_data),] #剩下的数据作为测试集
test_data <- OR[(train_size + 1):n]



#2.ARCH效应检验
#拟合AR(1)模型
ar_model<-arima(train_data,order=c(2,0,4))
print(ar_model)
# 获取残差
residuals<-residuals(ar_model)
#计算残差平方
residuals_squared<-residuals^2
#进行残差平方的相关性检验#Ljung-Box检验:
#原假设:不存在序列自相关
Box.test(residuals_squared,lag=10,type="Ljung")
Box.test(residuals_squared,lag=15,type="Ljung")
Box.test(residuals_squared,lag=20,type="Ljung")

#3.均值方差拟合
spec <- ugarchspec(variance.model =list(model ="sGARCH",garchOrder = c(1,1)),
                   mean.model=list(armaOrder=c(1,0),include.mean=TRUE))

fit<- ugarchfit(spec,data=train_data,out.sample = 0,solver = "gosolnp")
print(fit)



#4.模型预测
#模型的标准化残差分析(模型的验证)
#残差平方序列不具有序列相关性
resi<-residuals(fit,standardize=T)
Box.test(resi^2,lag=10,type="Ljung") #原假设不存在序列自相关


#模型预测
forecast <- ugarchforecast(fit,n.ahead =length(test_data))
#提取预测的波动率值
predicted_volatility <- forecast@forecast$seriesFor
# 计算预测精度指标，如均方误差(MSE)、平均绝对误差(MAE)、均方根误差(RMSE)
mse <-mean((predicted_volatility - test_data)^2)
mae <-mean(abs(predicted_volatility - test_data))
rmse<-sqrt(mean((predicted_volatility - test_data)^2))
#输出预测精度指标
print(paste("均方误差(MSE):", mse))
print(paste("平均绝对误差(MAE):",mae))
print(paste("均方根误差(RMSE):",rmse))