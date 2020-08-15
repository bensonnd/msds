title: "Case Study 2 - Talent Attrition and Income"  
author: "By Neil Benson"  
date: "08/05/2020"  

## Predicting attrition and income prepared for DDSAnalytics. 
In order to provide DDSAnalytics with a competetitive edge in Talent Management, we will optimize a classification Naive Bayes model to predict attrition first, and will finish up optimizing a linear regressio model to predict income. In the following explored and analyzed datasets provided by DDSAnalytics. odeling, used linear regression for initial variable selection, but trained and optimized the Naive Bayes classifation models using Naive Bayes.

## Importing the necessary libraries

* library(ggthemes)
* library(dplyr) 
* library(tidyverse)
* library(forcats)
* library(naniar)
* library(corrplot)
* library(imputeTS)
* library(e1071)
* library(caret)
* library(rpart)
* library(formula.tools)
* library(modelr)
* library(psych)
* library(gridExtra)
* library(ggplot2)
* library(grid)
* library(stringr)
* library(hash)
* library(tidyverse)

  
  
## Importing and Cleaning the Data
Data was cleaned and imputed as needed
 
  
## Overall Profile of Attritioned Employees
We've created an overall profile of employee churn. Attritioned employees are characterized by lower monthly income, younger, further from home, with fewer total working years, fewer years at company, fewer years in current role, slightly fewer years with current manager, and fewer stock options.

Profile of Attrition:  
  
  * Median Age: 32  
  * Department (most common): Research & Development  
  * Education Field (most common): Life Sciences  
  * Gender (most common): Men  
  * Job Role (most common): Sales Executive  
  * Marital Status (most common): Single  
  * Median Years at Company: 3  
  * Median Total Working Years: 6.5  
  * Median Monthly Income: $3,171 
  * Median Distance from Home: 9
  
  
## Eploring Job Role Specific Trends
|Job Role					|Avg Job Satisfaction	|Avg Monthly Income	|Avg Total Working Years	|Avg Years At Company	|Avg Years In Current Role	|Avg Years Since Last Promotion	|Avg Years With Curr Manager|
|---------------------------|-----------------------|-------------------|-----------------------|-------------------|-----------------------|-------------------------------|------------------------|
|Healthcare Representative	|2.83					|$7,435				|14						|9					|5						|3								|5                       |
|Human Resources			|2.56					|$3,285				|6						|5					|3						|1								|3                       |
|Laboratory Technician		|2.69					|$3,222				|8						|5					|3						|1								|3                       |
|Manager					|2.51					|$17,197			|25						|14					|6						|4								|6                       |
|Manufacturing Director		|2.72					|$7,505				|12						|8					|5						|2								|5                       |
|Research Director			|2.49					|$15,750			|21						|10					|6						|3								|6                       |
|Research Scientist			|2.80					|$3,259				|8						|5					|3						|2								|3                       |
|Sales Executive			|2.73					|$6,892				|11						|8					|5						|3								|5                       |
|Sales Representative		|2.70					|$2,653				|4						|3					|2						|1								|2                       |
  
  
## Checking the Balance of the Data
Because of the imbalance in employee churn, we will later downsample to balance the data to train our models.

  
## Exploring Colinearity in Numerical Variables  
Next we explored the colinearity of numerical values within the dataset.


## Transformations for Analysis
When analyzing colinearity, we noticed several attributes that could benefit from transformation, however upon further analysis, the transformations did not help in model building or optimization.
  
  
## Numerical Density by Attrition
When looking at the overlap of density, monthly hours, training time last year, and years since last promotion do not seem to uniquely identify individuals who might attrition, however may still be used to identify individuals in the model. More exploration is needed.

  
## Factor Analysis:
The top three most influential predictor factors in the dataset for attrition in accordance with an F-test are:  

* OverTime  
    * pvalue: < 8.45e-13  

* JobInvolvement  
    * pvalue: < 2.92e-06  

* JobLevel  
    * pvalue: 0.00108  
 
   
## Modeling and Optimization
Variable selection using forward, backward, and stepwise linear models. We have converted all factorial columns to numerical to proceed and compared the outputs of each of these models against the summary of the ANOVA table from reviewing top 3 factors.

Upon initial review of variable impacts in the three models, we have determined that at a minium we would like to include the following (as this selection is preliminary, these are most likely to change). Our p-value is an average of the 4 models analyzed - ANOVA of Factors, Forward LM, Backward LM, and Stepqise LMs  

|Predictor					        |Avg. p-value	  |
|---------------------------|---------------|
|OverTime					          |1.93252E-08   	|
|JobInvolvement				      |6.7925E-06	   	|
|TotalWorkingYears			    |0.000101467   	|
|YearsSinceLastPromotion	  |0.000228667  	|
|StockOptionLevel			      |0.00108    		|
|JobLevel					          |0.00108	     	|
|MaritalStatus				      |0.00190525	  	|
|JobSatisfaction			      |0.0049815   		|
|YearsWithCurrManager		    |0.006656	      |
|NumCompaniesWorked			    |0.009823667   	|  
  
  
Additionally, we would like to further investigate including the following variables. While they have p-value greater than .05, they are close and may hold significance with further testing:  
  
|Predictor					        |Avg. p-value 	|
|---------------------------|---------------|
|WorkLifeBalance		       	|0.0507765		  |
|RelationshipSatisfaction	  |0.06487075	  	|
|JobRole					          |0.06694	     	|  
  
  
And we will premilinarily move forward without the following variables, as they are currently statistically insignificant:  

|Predictor					        |Avg. p-value	  |
|---------------------------|---------------|
|Department					        |0.135287		    |
|HourlyRate					        |0.152496667  	|
|EducationField				      |0.2234		    	|
|EnvironmentSatisfaction	  |0.41002	    	|
|BusinessTravel				      |0.53436    		|
|PerformanceRating			    |0.54334     		|
|Education				        	|0.59587	     	|
|Gender					          	|0.9745	    		|  


## Further Refining the Model
We felt that the models from stepwise, forward and backward were not sufficient. We then added and subtracted columns to further refine the model.

## Fine Tuning the Models
The models we originally were working with were not meeting our specs, so we a function called modelOptimization to test various different combinations of predictor variables. We finally settled on a Naive Bayes classification model for Attrition before moving onto predicting income. For predicting income we created a function called modelPredictions that optimized variables in linear regression models to predict income. After many tests and optimizations we settled on a Naive Bayes classification model for Attrition and a Linear Regression model for income. In the end we ran them both against unlabeled data sets to score our models.

