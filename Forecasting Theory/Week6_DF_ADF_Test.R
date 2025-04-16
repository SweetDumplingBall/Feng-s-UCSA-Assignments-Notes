# 设定路径
setwd("E:/02_Postgraduate_UCAS/01_Class Tasks/01_Courses/02_预测理论与方法_2024秋_汤玲老师/03_Data")
# 加载包
library(zoo)
library(tseries)
library(readxl)
library(urca)
#导入数据
data <- read_excel("carbonprice.xlsx")
price = data[,2]

#frequency频次，start起始日期
price<- ts(price,frequency = 365, start = c(2024,2))

#df单位根检验 
#t统计量小于临界值的绝对值，接受原假设，检验结果是非平稳的
#type默认为trend，其余还有drift
ur.df(price)
summary(ur.df(price))
#adf单位根检验
result_adf_price <- adf.test(price)
print(result_adf_price)
#白噪声/纯随机检验 显著时候可认为是非白噪声序列
result_lb_price<-Box.test(price, type="Ljung-Box")
print(result_lb_price)



