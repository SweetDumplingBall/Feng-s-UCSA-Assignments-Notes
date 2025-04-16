
*******************************************************************************************************************
***************Assessing energy resilience and its greenhouse effect: A global perspective*************************
*******************************************************************************************************************

clear all
//use "E:\02_Postgraduate_UCAS\01_Class Tasks\01_Courses\02_预测理论与方法_2024秋_汤玲老师\03_Data\联立方程案例.dta", clear
gen lnco2=log(co2)   //Natural logarithm processing for each variable
gen lneri=log(eri)   //Natural logarithm processing for each variable
gen lneai=log(eai)   //Natural logarithm processing for each variable
gen lnrei=log(rei)   //Natural logarithm processing for each variable
gen lneei=log(eei)   //Natural logarithm processing for each variable
gen lnpgdp=log(pgdp) //Natural logarithm processing for each variable
gen lnpgdp2=lnpgdp*lnpgdp   //Natural logarithm processing for each variable
gen lngdp=log(gdp)   //Natural logarithm processing for each variable
gen lnisa=log(isa)   //Natural logarithm processing for each variable
gen lntra=log(tra)   //Natural logarithm processing for each variable
gen lnlab=log(lab)   //Natural logarithm processing for each variable
gen lnurb=log(urb)   //Natural logarithm processing for each variable

***************************************************************
******Table 8 Mechanism analysis for energ resilience**********
***************************************************************
gen tech=co2/gdp
gen lntech=log(tech)
gen lnscale=log(gdp)
gen lncom=log(isa)
reg3 (lnscale lneri lnpgdp lntra lnlab lnurb) (lntech lneri lnpgdp lnpgdp2 lncom) (lncom lneri lnscale lntra lnlab), 3sls 
export excel using filename, sheet("Sheet1") firstrow(variables) replace