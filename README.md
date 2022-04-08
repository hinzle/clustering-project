# Clustering Project

### codeup/innis - 2020 mar 30

---
 
## Table of Contents

 
## 1. Objective : 
Use clustering and regresion to determine drinvers of "Zestimate".
"We want to be able to predict the property values ('logerror') of Single Family Properties that had a transaction during 2017."  
> https://ds.codeup.com/clustering/project/


## 2. Dataset : Zillow  

- ### Description: 

	properties_2017.csv - all the properties with their home features for 2017 (released on 10/2/2017)

- ### Profile :

	"Zillow’s Zestimate home valuation has shaken up the U.S. real estate industry since first released 11 years ago.

	A home is often the largest and most expensive purchase a person makes in his or her lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. The Zestimate was created to give consumers as much information as possible about homes and the housing market, marking the first time consumers had access to this type of home value information at no cost.

	“Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning."

	> https://www.kaggle.com/competitions/zillow-prize-1/data

 

## 3. Initial Questions:


- Why do some properties have a much higher value than others when they are located so close to each other? 
- Does sqaure footage effect property value? 
- Does number of baths effect property value?
- Does number of beds effect property value?
- What is the optimal ratio of beds/baths?


##### Data-Focused Questions
- [x] Is there a relationship between logerror and lot size in each county?
- [x] Is there a relationship between logerror and finished square footage of the property in each county?
- [x] Is logerror significantly different for properties in LA County vs Orange County vs Ventura County?
- [x] Is there a relationship between logerror and zipcode?
- [x] Controlling for property square footage, what is the relationship between logerror and age of the home?

 
##### Overall Project-Focused Questions

- What will the end product look like?
- What format will it be in?
- Who will it be delivered to?
- How will it be used?
- How will I know I'm done?
- What is my MVP?
- How will I know it's good enough?
 

#### 4. FORMULATING HYPOTHESES
The project started with the hypothesis that the lot size, finished square feet, and location of the property would have the greatest impact on the residual error.  

 
#### 5. DELIVERABLES:
- [x] Github Repo - containing a final report (.ipynb), acquire & prepare modules in the form of a Wrangle.py, as well as supplemental files such as imports.py.
- [x] README file - provides an overview of the project and steps for project reproduction. 
- [x] Draft Jupyter Notebooks - provide all steps taken to produce the project.
- [x] Python Module File - provides reproducible code for acquiring and preparing the data.
- [x] wrangle.py - used to acquire data and prepare data
- [x] Report Jupyter Notebook - provides final presentation-ready assessment and recommendations.
- [x] 5 minute presentation to stakeholders (Zillow Data Science Team). 
 
 
## II. PROJECT DATA CONTEXT
 
#### 1. DATA DICTIONARY:
The final DataFrame used to explore the data for this project contains the following variables (columns).  The variables, along with their data types, are defined below in alphabetical order:
 

| Variable                       | Definition                                         | Data Type |
|:-------------------------------|:--------------------------------------------------:|:---------:|
| acres                          | grouped bins, based upon lot square footage        | category  |
| age                            | grouped bins, based upon year built.               | category  |
| assessed_value                 | total tax assessed value of the property           | float64   |
| bathroom_bins                  | grouped bins, based upon number of bedrooms        | category  |
| bathrooms                      | number of bathrooms and half-bathrooms in home     | float64   |
| bedroom_bins                   | grouped bins, based upon number of bedrooms        | category  |
| bedrooms                       | number of bedrooms in the home                     | float64   |
| county_code_bin                | name of county as assigned by state_county_code    | category  |
| county_code_bin_Orange County  | numeric variable representing county_code_bin      | uint8     |
| county_code_bin_Ventura County | numeric variable representing county_code_bin      | uint8     |
| latitude                       | Latitude of the middle of the parcel multiplied by 10e6  | category  |
| logerror                       | Residual Error in Home Valuation                   | float64   |
| longitude                      | Longitude of the middle of the parcel multiplied by 10e6 | category  |
| home_sizes                     | grouped bins, based upon square footage            | category  |
| square_feet                    | total finished living area of the home             | float64   |
| state_county_code              | federal information processing standards code      | object    |
| total_rooms                    | combined number of bedrooms and bathrooms          | float64   |
| year_built                     | year the primary residence was constructed         | int64     |




## III. PROJECT PLAN - USING THE DATA SCIENCE PIPELINE:
The following outlines the process taken through the Data Science Pipeline to complete this project. 
 
Plan➜ Acquire ➜ Prepare ➜ Explore ➜ Model & Evaluate ➜ Deliver
 
#### 1. PLAN
- [x] Review project expectations
- [x] Draft project goal to include measures of success
- [x] Create questions related to the project
- [x] Create questions related to the data
- [x] Create a plan for completing the project using the data science pipeline
- [x] Create a data dictionary to define variables and data context
- [x] Draft starting hypothesis
 
#### 2. ACQUIRE
- [x]  Create .gitignore
- [x]  Create env file with log-in credentials
- [x]  Store env file in .gitignore to ensure the security of sensitive data
- [x]  Create wrangle.py module
- [x]  Store functions needed to acquire the Zillow dataset from mySQL
- [x]  Ensure all imports needed to run the functions are inside the imports.py document
- [x]  Using Jupyter Notebook
- [x]  Run all required imports
- [x]  Import functions from aquire.py module
- [x]  Summarize dataset using methods and document observations
 
#### 3. PREPARE
Using Jupyter Lab Notebook
- [x] Create prepare functions in the wrangle.py module
- [x] Store functions needed to prepare the Zillow data such as:
   - [x] Split Function: to split data into train, validate, and test
   - [x] Cleaning Function: to clean data for exploration
   - [x] Encoding Function: to create numeric columns for object column
   - [x] Feature Engineering Function: to create new features
- [x] Ensure all imports needed to run the functions are inside the wrangle.py document
Using Jupyter Notebook
- [x] Import functions from prepare.py module
- [x] Summarize dataset using methods and document observations
- [x] Clean data
- [x] Features need to be turned into numbers
- [x] Categorical features or discrete features need to be numbers that represent those categories
- [x] Continuous features may need to be standardized to compare like datatypes
- [x] Address missing values, data errors, unnecessary data, renaming
- [x] Split data into train, validate, and test samples
 
#### 4. EXPLORE
Using Jupyter Notebook:
- [x] Answer key questions about hypotheses and find drivers of residual error
  - Run at least two statistical tests
  - Document findings
- [x] Create visualizations with the intent to discover variable relationships
  - Identify variables related to property values
  - Identify any potential data integrity issues
- [x] Summarize conclusions, provide clear answers, and summarize takeaways
  - Explain plan of action as deduced from work to this point
 
#### 5. MODEL & EVALUATE
Using Jupyter Notebook:
- [x] Establish baseline accuracy
- [x] Train and fit multiple (3+) models with varying algorithms and/or hyperparameters
- [x] Compare evaluation metrics across models
- [x] Remove unnecessary features
- [x] Evaluate best performing models using validate set
- [x] Choose best performing validation model for use on test set
- [x] Test final model on the out-of-sample testing dataset
- [x] Summarize performance
- [x] Interpret and document findings
 
#### 6. DELIVERY
- [x] Prepare five-minute presentation using Jupyter Notebook
- [x] Include an introduction of project and goals
- [x] Provide an executive summary of findings, key takeaways, and recommendations
- [x] Create walkthrough of analysis 
  - Visualize relationships
  - Document takeaways
  - Explicitly define questions asked during the initial analysis
- [x] Provide final takeaways, recommend a course of action, and next steps
- [x] Be prepared to answer questions following the presentation

 
 
## IV. PROJECT MODULES:
- [x] Python Module Files - provide reproducible code for acquiring,  preparing, exploring, & testing the data.
   - [x] imports.py - used to store all imports needed to run functions and processes
   - [x] wrangle.py - used to acquire and prepare data
 
  
## V. PROJECT REPRODUCTION:
### Steps to Reproduce
 
- [x] You will need an env.py file that contains the hostname, username, and password of the mySQL database that contains the zillow database
- [x] Store that env file locally in the repository
- [x] Make .gitignore and confirm .gitignore is hiding your env.py file
- [x] Clone my repo (including the imports.py and wrangle.py)
- [x] Import python libraries:  pandas, matplotlib, seaborn, numpy, and sklearn
- [x] Follow steps as outlined in the README.md. and workbook.ipynb
- [x] Run final_report.ipynb to view the final product