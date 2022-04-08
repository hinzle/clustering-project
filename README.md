# Clustering Project

### codeup/innis - 2020 mar 30

---
 
## Table of Contents

 
## I. Objective : 
Use clustering and regresion to determine drinvers of "Zestimate".
"We want to be able to predict the property values ('logerror') of Single Family Properties that had a transaction during 2017."  
> https://ds.codeup.com/clustering/project/


## II. Dataset : Zillow  

- ### Description: 

	properties_2017.csv - all the properties with their home features for 2017 (released on 10/2/2017)

- ### Profile :

	"Zillow’s Zestimate home valuation has shaken up the U.S. real estate industry since first released 11 years ago.

	A home is often the largest and most expensive purchase a person makes in his or her lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. The Zestimate was created to give consumers as much information as possible about homes and the housing market, marking the first time consumers had access to this type of home value information at no cost.

	“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning."

	> https://www.kaggle.com/competitions/zillow-prize-1/data

 

## III. Initial Questions:


- Why do some properties have a much higher value than others when they are located so close to each other? 
- Does sqaure footage effect property value? 
- Does number of baths effect property value?
- Does number of beds effect property value?
- What is the optimal ratio of beds/baths?


## IV. Data Dictionary

| Variable | Description |
|---|---|
|'beds'| Number of bedrooms in home |
|'baths'| Number of bathrooms in home including fractional bathrooms|
|'sqft'| Calculated total finished living area of the home |
|'fips'| Federal Information Processing Standard code -  see https://en.wikipedia.org/wiki/FIPS_county_code for more details|
|'year'| The Year the principal residence was built |
|'taxes'|The total property tax assessed for that assessment year|
|'property_value'|The total tax assessed value of the parcel|