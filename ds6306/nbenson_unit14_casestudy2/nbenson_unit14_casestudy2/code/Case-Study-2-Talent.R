library(ggplot2) 
library(ggthemes)
library(dplyr) 
library(tidyverse)
library(forcats)

talent <- read.csv("C:/Git/MSDS_6306_Doing-Data-Science/Unit 14 and 15 Case Study 2/CaseStudy2-data.csv")


attrittioned <- talent %>% filter(Attrition == "Yes")
nonattrittioned <- talent %>% filter(Attrition == "No")

summary(attrittioned)
summary(nonattrittioned)


# function to summarize and profile those who have attritioned and those who have not
profiles <- function (dataframe) {
  dfnames <- names(dataframe)
  outputdf <- data.frame()[, ]
  
  for(name in dfnames){
  
    if (class(dataframe$name) == 'integer')
      { 
        outputdf[name,] <- mean(dataframe$name)
      } 
    else
      { 
        outputdf[name,] <- mode(dataframe$name)
      }
  }
  return(outputdf)
}

attrtn_profile <- profiles(attrittioned)

