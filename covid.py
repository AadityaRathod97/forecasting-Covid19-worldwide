

import pandas as pd
import numpy as np
covid = pd.read_excel("covid.xlsx") #read csv file

Train = covid.head(64)
Test = covid.tail(8)


####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('Confirmed~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
MAPE_linear = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_linear))/Test['Confirmed']))*100
MAPE_linear #Mean Absolute Percentage Error



##################### Exponential ##############################

Exp = smf.ols('logconfirmed~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
MAPE_Exp = np.mean(abs((np.array(Test['Confirmed'])-np.array(np.exp(pred_Exp)))/Test['Confirmed']))*100
MAPE_Exp




#################### Quadratic ###############################

Quad = smf.ols('Confirmed~t+t2',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t2"]]))
MAPE_Quad = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_Quad))/Test['Confirmed']))*100
MAPE_Quad


################### Linear trend + Daily Seasonality####################
lintrendea = smf.ols('Confirmed~t+sin+cos',data=Train).fit()
pred_lintrendea = pd.Series(lintrendea.predict(Test[["t","sin","cos"]]))
MAPE_lintrendea = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_lintrendea))/Test['Confirmed']))*100
MAPE_lintrendea

##################### Exponential Trend + Daily seasonality #################

exptrendea = smf.ols('logconfirmed~t+sin+cos',data=Train).fit()
pred_exptrendea = pd.Series(exptrendea.predict(Test[["t","sin","cos"]]))
MAPE_exptrendea = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_exptrendea))/Test['Confirmed']))*100
MAPE_exptrendea


################### Linear trend + Daily Seasonality + lag ####################
lintrendealag = smf.ols('Confirmed~t+lagconfirmed+sin+cos',data=Train).fit()
pred_lintrendealag = pd.Series(lintrendealag.predict(Test[["t","sin","cos","lagconfirmed"]]))
MAPE_lintrendealag = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_lintrendealag))/Test['Confirmed']))*100
MAPE_lintrendealag




################### Additive seasonality ########################

add_sea = smf.ols('Confirmed~lagconfirmed',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['lagconfirmed']]))
MAPE_add_sea = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_add_sea))/Test['Confirmed']))*100
MAPE_add_sea



################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('Confirmed~t+t2+lagconfirmed',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['lagconfirmed','t','t2']]))
MAPE_add_sea_quad = np.mean(abs((np.array(Test['Confirmed'])-np.array(pred_add_sea_quad))/Test['Confirmed']))*100
MAPE_add_sea_quad 


################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('logconfirmed~lagconfirmed',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
MAPE_Mult_sea = np.mean(abs((np.array(Test['Confirmed'])-np.array(np.exp(pred_Mult_sea)))/Test['Confirmed']))*100
MAPE_Mult_sea


##################Multiplicative Additive Seasonality ###########

Mul_Add_sea = smf.ols('logconfirmed~t+lagconfirmed',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
MAPE_Mult_add_sea = np.mean(abs((np.array(Test['Confirmed'])-np.array(np.exp(pred_Mult_add_sea)))/Test['Confirmed']))*100
MAPE_Mult_add_sea 



################## Testing #######################################

data = {"MODEL":pd.Series(["MAPE_Exp","MAPE_Mult_add_sea","MAPE_Mult_sea","MAPE_Quad","MAPE_add_sea","MAPE_add_sea_quad","MAPE_exptrendea","MAPE_linear","MAPE_lintrendea","MAPE_lintrendealag"]),"RMSE_Values":pd.Series([MAPE_Exp,MAPE_Mult_add_sea,MAPE_Mult_sea,MAPE_Quad,MAPE_add_sea,MAPE_add_sea_quad,MAPE_exptrendea,MAPE_linear,MAPE_lintrendea,MAPE_lintrendealag])}
table_MAPE=pd.DataFrame(data)
table_MAPE
# so rmse_add_sea_quad has the least value among the models prepared so far


 



# Predicting new values 

predict_data = pd.read_csv("Predict_new.csv")
model_full = smf.ols('Confirmed~t+t2+lagconfirmed',data=covid).fit()

pred_new  = pd.Series(lintrendealag.predict(predict_data))

pred_new
predict_data["forecasted_Confirmed"] = pd.Series(pred_new)
