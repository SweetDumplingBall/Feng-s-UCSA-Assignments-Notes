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


#数据集划分
n <- nrow(data)
train_size <- round(0.8*n) #计算训练集大小，取前80%的数据
train_data <- data[1:train_size, ] #前80%数据作为训练集
#test_data <- carbon_data[(train_size + 1):nrow(carbon_data),] #剩下的数据作为测试集
test_data <- data[(train_size + 1):n, ]

#根据信息准则确定ARDL模型
# 在训练集循环不同的滞后阶数，并计算AIC
best_aic <- Inf
best_p <- best_q <- NULL

# 在训练集循环不同的滞后阶数，并计算AIC
for (p in 1:3) {
  for (q in 1:3) {
    # 复制train_data以避免在原始数据上进行修改
    temp_data <- train_data
    # 生成Y的滞后变量
    for (i in 1:p) {
      col_name <- paste0("y_lag", i)
      temp_data[[col_name]] <- lag(temp_data$y, i)
    }
    # 生成X的滞后变量
    for (i in 1:q) {
      col_name <- paste0("x_lag", i)
      temp_data[[col_name]] <- lag(temp_data$x, i)
    }
    # 去除因生成滞后变量而产生的NA值
    temp_data <- na.omit(temp_data)
    # 构建模型公式
    formula_parts <- c("y")
    for (i in 1:q) {
      formula_parts <- c(formula_parts, paste0("x_lag", i))
    }
    for (i in 1:p) {
      formula_parts <- c(formula_parts, paste0("y_lag", i))
    }
    formula <- as.formula(paste("y ~", paste(formula_parts, collapse = " + ")))
    # 拟合线性模型
    model <- lm(formula, data = temp_data)
    # 获取模型的AIC值
    aic <- AIC(model)
    # 如果当前模型的AIC值小于已知的最小AIC值，则更新最佳模型参数
    if (aic < best_aic) {
      best_aic <- aic
      best_p <- p
      best_q <- q
    }
  }
}

# 输出最佳滞后阶数和对应的AIC值
cat("最佳滞后阶数为X的", best_q, "阶和Y的", best_p, "阶，对应的AIC值为", best_aic, "\n")

# 使用最佳滞后阶数best_p和best_q拟合最终模型
final_model_data <- train_data
for (i in 1:best_p) {
  col_name <- paste0("y_lag", i)
  final_model_data[[col_name]] <- lag(final_model_data$y, i)
}
for (i in 1:best_q) {
  col_name <- paste0("x_lag", i)
  final_model_data[[col_name]] <- lag(final_model_data$x, i)
}
final_model_data <- na.omit(final_model_data)

# 构建模型公式
formula_parts <- c("x") # 开始时包含x
for (i in 1:best_q) {
  formula_parts <- c(formula_parts, paste0("x_lag", i))
}
for (i in 1:best_p) {
  formula_parts <- c(formula_parts, paste0("y_lag", i))
}
final_formula <- as.formula(paste("y ~", paste(formula_parts, collapse = " + ")))

# 拟合最终模型
final_model <- lm(final_formula, data = final_model_data)
summary(final_model)

# 模型的残差分析(模型的验证)
resi <- residuals(final_model, standardize = TRUE)
Box.test(resi, lag = 10, type = "Ljung") # 原假设不存在序列自相关

# 生成测试集的滞后变量
for (i in 1:best_q) {
  col_name <- paste0("x_lag", i)
  test_data[[col_name]] <- lag(test_data$x, i)
}
for (i in 1:best_p) {
  col_name <- paste0("y_lag", i)
  test_data[[col_name]] <- lag(test_data$y, i)
}
test_data <- na.omit(test_data)

# 预测数据
predicted_values <- predict(final_model, newdata = test_data)
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

#--------------------------------------------------------------------------------------------------
#以下内容暂时无用
#构建线性回归模型,DL(0)
model <- lm(y ~.,data=train_data)
summary(model)

# DL(1) 和上面一样，因为生成的滞后项已经加入x中
model <- lm(y ~x+lag_1+lag_2,data=train_data)
summary(model)
#获取模型信息准则
aic <- AIC(model)
bic <- BIC(model)
print(aic)
print(bic)

#生成滞后项
data$lag_1 <- lag(data$x,1)
data$lag_2 <- lag(data$x,2)
#去除含有NA值的观测
data <- na.exclude(data)