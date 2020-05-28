if (!require(BatchGetSymbols)) install.packages('BatchGetSymbols')
library(xts)
library(BatchGetSymbols)
library(ggplot2)
library(fpp)  # forecasting principles and practices 
library(prophet)  # facebook's forecasting package 
library(janitor)  # for cleaning up dataframes  
library(lubridate)  # for working with dates 
library(tidyr)  # for tidying data
library(forecast)
library(rugarch)

n = 30 #number of predict dates
# set dates
first.date <- Sys.Date() - 2000
last.date <- Sys.Date()
freq.data <- 'daily'
# set tickers
tickers <- c('FB')

l.out <- BatchGetSymbols(tickers = tickers, 
                         first.date = first.date,
                         last.date = last.date, 
                         freq.data = freq.data,
                         cache.folder = file.path(tempdir(), 
                                                  'BGS_Cache') ) # cache in tempdir()


print(l.out$df.control)

#View(l.out$df.tickers)

# p <- ggplot(l.out$df.tickers, aes(x = ref.date, y = price.close))
# p <- p + geom_line()
# p <- p + facet_wrap(~ticker, scales = 'free_y')
# print(p)

#Create FB stock time series
fb_ts <- xts(l.out$df.tickers$price.close, as.Date(l.out$df.tickers$ref.date, format='%Y-%m/%d'))
#summary(fb_ts)

#plot time series data using autoplot, can also use ggplot
autoplot(fb_ts) +
  ggtitle("Facebook daily stock price") +
  xlab("Year") +
  ylab("Closing sock price")

#Plotting autocorrelation, indicating strong correlation for long lag periods
ggAcf(fb_ts, lag=50) #Exponential decay to zero indicates autoregressive model 
ggPacf(fb_ts, lag=50)



#Simple forecast of time series using naive(random walk), mean and seasonal naive method, rwf with drift
fb2 <- window(fb_ts, start=first.date, end=last.date-n) #Training data

fb3 = window(fb_ts, start=last.date-n) #Test data

#test dates sequence
dates <- seq(as.Date(last.date-n+1, format = "%Y-%m-%d"),
             as.Date(last.date-1, format = "%Y-%m-%d"),
             by = 1 )

dates <- dates[!weekdays(dates) %in% c('Saturday','Sunday')]

length(dates)
length(fb3)

fb_fit1 <- meanf(fb2, h=length(dates))
fb_fit2 <- rwf(fb2, h=length(dates))
fb_fit3 <- rwf(fb2, drift=TRUE, h=length(dates))


summary(fb_fit3) #getting summary of fit using means forecasting method

print(fb_fit3$mean)

plot(fb_ts, main="FB daily stock price", ylab="Closing price", xlab="Years")

#Results with mean and random walk with drift method don't look good. Let us check the accracy with test data 
accuracy(fb_fit1, fb3) #accuracy of mean method
accuracy(fb_fit2, fb3) #accuracy of naive method
accuracy(fb_fit3, fb3) #accuracy of random walk with drift 


#Let us now try ARIMA method for time series forecasting. Note the above methods are actually just special cases of ARIMA. ARIMA tries to find optimal p, d, q parameters to fit the time series

#Conduct Augmented Dickey Fuller test
print(adf.test(fb_ts)) #non-stationary indication, since we are not able to reject null hypothesis of non-stationary time series

fb_arima <- auto.arima(fb2, lambda = "auto", seasonal=FALSE)
summary(fb_arima)
fb_arima_forecast <- forecast(fb_arima,h=length(dates))
plot(fb_arima_forecast)
#View(fb_arima_forecast)
accuracy(fb_arima_forecast, fb3) #RMSE is lower than mean, random walk methods 

#Residual time series plot, no particular pattern observed so a good fit
res <- residuals(fb_arima)
autoplot(res) + xlab("Day") + ylab("") +
  ggtitle("Residuals from ARIMA method")

#ACF plot of residuals
ggAcf(res, lag=50) #most values should be inside blue dotten lines, indicating no significant autocorrelation


#Box test for lag=2
Box.test(res, lag= 2, type="Ljung-Box")

#general Box test for 
Box.test(res, type="Ljung-Box") #high p-values indicate we are not able to reject the null hypothesis that residuals are indicatively iid

qqnorm(res, main="Normal Q-Q Plot ARIMA")
qqline(res)





#Next do ETS and a comparison with ARIMA 



#Use GARCH 
#Dataset forecast upper first 5 values
fitarfima = autoarfima(data = fb2, ar.max = 2, ma.max = 2, 
                       criterion = "AIC", method = "full")


#define the model
garch_model=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(1,2)))
#estimate model 
garch_fit=ugarchfit(spec=garch_model, data=fb2)

#conditional volatility plot
plot.ts(sigma(garch_fit), ylab="sigma(t)", col="blue", main="GARCH conditional volatility")

#Model akike
infocriteria(garch_fit)


#Normal residuals
garchres <- data.frame(residuals(garch_fit))  
plot(garchres$residuals.garch_fit., main="GARCH residuals after fitting")


#Standardized residuals
garchres <- data.frame(residuals(garch_fit, standardize=TRUE)) 
#Normal QQ plot
qqnorm(garchres$residuals.garch_fit..standardize...TRUE., main="Normal Q-Q Plot GARCH")
qqline(garchres$residuals.garch_fit..standardize...TRUE.)

#Squared standardized residuals Ljung Box
garchres <- data.frame(residuals(garch_fit, standardize=TRUE)^2) 
Box.test(garchres$residuals.garch_fit..standardize...TRUE..2, type="Ljung-Box")

#GARCH Forecasting
garchforecast <- ugarchforecast(garch_fit, n.ahead = length(dates) )
show(garchforecast)
#plot(garchforecast)
#Use Prophet




#Use neural network



#Plots for predictions
plot(dates, fb3, main="Predictions vs actual stock price", xlab="Dates",
     ylab="Close price",  col="black", type='l', ylim=c(100, 300)) 
lines(dates, fb_fit3$mean, col="red")
# Plotting the test set
lines(dates, fb_fit1$mean, col="blue")
lines(dates, as.vector(fb_arima_forecast$mean), col="green")

# legend
legend("topleft", lty=1, col=c( "green", "blue" , "red", "black"),
       legend=c( "ARIMA", "Mean method","Random walk with drift", "Actual stock price"),bty="n")



