#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#https://www.kaggle.com/shinjinibasu/summer-olympics-exploration-and-linear-regression
"""
Created on Sat Oct 17 23:45:39 2020

@author: Mansi Sethi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns #good looking statistical data visualization 
import plotly.graph_objects as go

import os
os.getcwd()
os.chdir('/Users/raghav//Desktop/documents/courses ba cu denver/bana 6620 ba computing/project_olympics')

#GETTING BOTH THE NEEDED DATA SETS:
OlympicsData = pd.read_csv('athlete_events.csv')
GDPdata=pd.read_csv('GDP.csv', skiprows=4) #skipping 4 rows with extra info
GDPdata.head()
GDPdata.describe

#working on olympics data first
OlympicsData.head(5) #looking at data

OlympicsData.describe() #basic stats
OlympicsData.info() #provides count, type of variable etc

#removing duplicates #based on entirety of row duplicate
OlympicsDat=OlympicsData.drop_duplicates()
OlympicsDat.info()
OlympicsDat.shape #(269731, 15) #original file before removal of duplicates had 271116 rows


#print out all column names
for colName in OlympicsDat:
    print(colName)
 
#Printing columns with Null values
print(OlympicsDat.isnull().sum())

#looking at missing values visually:
pip install missingno
import missingno
missingno.matrix(OlympicsDat)
 

#REPLACING BLANKS FOR MEDALS WITH 'No Medal'
OlympicsDat['Medal'].fillna('No Medal',inplace=True ) 

#Looking at empty columns again:
OlympicsData.isnull().sum()
  
#keeping only needed columns 
#dropping ID, Name, Games, city   
cats = ["Sex","Age","Height","Weight","Team","NOC","Year","Season","Sport","Medal"]

OlympicsDat = OlympicsDat.loc[:,"cats"] 

OlympicsDat.head(5)
OlympicsDat.describe() 

#no of unique events
Count_events=OlympicsDat.groupby(by=['Year','Season'],as_index=False)['Sport'].count()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Start of Mike's analysis



Questions?
What did we decide to do with NaN values in height, weight?
Should we have consistent colors for Seaborn model

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.pyplot as plt
import seaborn as sns #good looking statistical data visualization 
import plotly.graph_objects as go
from plotly.offline import plot 

OlympicsData = pd.read_csv('athlete_events.csv')


#removing duplicates #based on entirety of row duplicate
OlympicsDat=OlympicsData.drop_duplicates()


#REPLACING BLANKS FOR MEDALS WITH 'No Medal'
OlympicsDat['Medal'].fillna('No Medal',inplace=True ) 


#keeping only needed columns 
#dropping ID, Name, Games, city   
cats = ["Sex","Age","Height","Weight","Team","NOC","Year","Season","Sport","Medal"]

OlympicsDat = OlympicsDat.loc[:,cats] 


"""

Beginning of Mike's analysis

"""

#create objects only for the summer games
summer = OlympicsDat[(OlympicsDat.Season == 'Summer')]


#Histograms for distribution of age, weight and height across the summer games

#Age
plt.hist(summer['Age'], bins = 40, histtype = 'bar', ec = 'black')
plt.axvline(summer['Age'].mean(), color = 'k', linestyle = 'dashed', linewidth = 1)
#add a black dashed line per the mean age
plt.xlabel('Age')
plt.title('Age Distribution\n Summer Games')
plt.figtext(0.7,0.5,summer['Age'].describe().astype(int).to_string())
#Place the descriptive statistics on the histogram. 
plt.text(summer['Age'].mean()*1.1, max_ylim*0.9, 'Mean: {:.2f}'.format(summer['Age'].mean()))
#text with location formating to place the mean of age near the dashed line


#Weight
plt.hist(summer['Weight'], bins = 40, histtype = 'bar', ec = 'black')
plt.axvline(summer['Weight'].mean(), color = 'k', linestyle = 'dashed', linewidth = 1
#add a black dashed line per the mean weight
plt.xlabel('Weight, kg')
plt.title('Weight Distribution\n Summer Games')
plt.figtext(0.6,0.5,summer['Weight'].describe().astype(int).to_string())
#Place the descriptive statistics on the histogram. 
plt.text(summer['Weight'].mean()*1.1, max_ylim*0.43, 'Mean: {:.2f}'.format(summer['Weight'].mean())) 
#text with location formating to place the mean of weight near the dashed line


#Height
plt.hist(summer['Height'], bins = 40, histtype = 'bar', ec = 'black')
plt.axvline(summer['Height'].mean(), color = 'k', linestyle = 'dashed', linewidth = 1)
#add a black dashed line per the mean height
plt.xlabel('Height, cm')
plt.title('Height Distribution\n Summer Games')
plt.figtext(0.16,0.5,summer['Height'].describe().round(2).to_string())
#Place the descriptive statistics on the histogram. 
plt.text(summer['Height'].mean()*1.01, max_ylim*0.32, 'Mean: {:.2f}'.format(summer['Height'].mean()))
#text with location formating to place the mean of height near the dashed line


### Which countries have the best records for medals won?
#Make an object that only contains countries that have won medals
medals = summer.loc[summer['Medal'] != 'No Medal']

#To find the top 5 medal winners, we use
medals.Team.value_counts().head(5)


#We see that the top 5 countries that won medals are the US, USSR, DE, GB, FR
#we'll only keep those winning countries

countries = ["United States","Soviet Union","Germany","Great Britain","France"]
topMedalWinners = medals[medals['Team'].isin(countries)]


### To display the winnings by country, a stacked bar plot with custom colors to match
#medal colors was created
fig = go.Figure() #define a figure
fig.add_bar(name="Bronze", x=countries, y=topMedalWinners[topMedalWinners.Medal == "Bronze"].
            Team.value_counts().reindex(countries), marker_color="#A57164", width = 0.75)
#add bars to the figure you defined (fig). count the number of medals and group by country ('Teams')
fig.add_bar(name="Silver", x=countries, y=topMedalWinners[topMedalWinners.Medal == "Silver"].
            Team.value_counts().reindex(countries), marker_color="#C0C0C0", width = 0.75)
fig.add_bar(name="Gold", x=countries, y=topMedalWinners[topMedalWinners.Medal == "Gold"].
            Team.value_counts().reindex(countries), marker_color="#FFD700", width = 0.75)

fig.update_layout(
    title = 'Top 5 Countries for Medal Wins',
    barmode='stack', 
    xaxis = dict(categoryorder = 'total descending', tickangle = -45,
    ))
#define the chart as a stacked bar chart, sort the number of medals won by country, angle lables for country name
plot(fig)



#medal destribution by age and gender
sns.boxplot(x = 'Medal', y = 'Age', hue = 'Sex', order = ['Bronze' , 'Silver', 'Gold' ] , saturation=1, data = medals) 
plt.xlabel('Medal Awarded', fontsize = 12)
plt.ylabel('Age', fontsize = 12)
plt.title('Medals Awarded, by Gender and Age' , fontsize = 14)

#from the boxplot, we can see that there were some young winners around 10 years of age and some into the 70s
#let's see what gold medal winners were under 15 and over 60 

goldMedals = medals.loc[medals['Medal'] == 'Gold']#create object that is only gold medal winners

#drill down the data to see all the gold medal winners
plt.hist(goldMedals['Age'], bins = 40, histtype = 'bar', ec = 'black')
plt.axvline(goldMedals['Age'].mean(), color = 'k', linestyle = 'dashed', linewidth = 1)
plt.xlabel('Age')
plt.title('Age Distribution\n Gold Medal Winners')
plt.figtext(0.7,0.5,goldMedals['Age'].describe().astype(int).to_string())
plt.text(goldMedals['Age'].mean()*1.05, max_ylim*0.035, 'Mean: {:.2f}'.format(goldMedals['Age'].mean()))


#There were some very young (arond 10 years old) and very old (around 70 Years old) winners
###  What events did young and old win medals in?

goldMedals15 = goldMedals[(goldMedals.Age <= 15)]
plt.figure(figsize=(8, 5))
g = sns.countplot(y= 'Sport', hue = 'Sex', data = goldMedals15, order=(goldMedals15['Sport'].value_counts().index))
plt.xlabel('Count', fontsize = 12)
plt.ylabel('Sport', fontsize = 12)
g.legend(loc='center left', bbox_to_anchor=(0.85, 0.5), title = 'Sex')
plt.xticks( fontsize = 10)
plt.yticks(rotation = 45, fontsize = 10)
plt.title('Gold Medals for Athletes 15 and Younger', fontsize = 16)



goldMedals50 = goldMedals[(goldMedals.Age >= 50)]
plt.figure(figsize=(8, 5))
g = sns.countplot(y= 'Sport', hue = 'Sex', data = goldMedals50, order=(goldMedals50['Sport'].value_counts().index))
plt.xlabel('Count', fontsize = 12)
plt.ylabel('Sport', fontsize = 12)
g.legend(loc='center left', bbox_to_anchor=(0.85, 0.5), title = 'Sex')
plt.xticks( fontsize = 10)
plt.yticks(rotation = 45, fontsize = 10)
plt.title('Gold Medals for Athletes 50 and Older', fontsize = 16)

###Roque was something the older athletes won. What is Roque?
#### Is roque something that is still played at the Olympics?

#define an object that focuses on Sports that only contain roque and croquet
#the focus of data was from the object summer, which was all of the original data but only for summer
#This way we don't only see roque and croquet winners, but all participant
roqueAndCroquet = ["Roque","Croquet"]
croquetAndRoqueOverYears = summer[summer['Sport'].isin(roqueAndCroquet)]

#bar chart to show the years and amount of oeioke who played
sns.countplot(x='Year', data=croquetAndRoqueOverYears, hue = 'Sport')
plt.title('Croquet and Roque over Time')


#medal destribution by height and gender
plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Medal', y = 'Height', hue = 'Sex', order = ['Bronze' , 'Silver', 'Gold' ] , saturation=1 , data = medals) 
plt.xticks(rotation = 45)
plt.xlabel('Medal Awarded', fontsize = 12)
plt.ylabel('Height in cm', fontsize = 12)
plt.title('Medals Awarded, by Gender and Height' , fontsize = 14)


#medal destribution by weight and gender
plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Medal', y = 'Weight', hue = 'Sex', order = ['Bronze' , 'Silver', 'Gold' ] , saturation=1 , data = medals) 
plt.xticks(rotation = 45)
plt.xlabel('Medal Awarded', fontsize = 12)
plt.ylabel('Height in cm', fontsize = 12)
plt.title('Medals Awarded, by Gender and Weight' , fontsize = 14)


# Height and Gender by year
plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Year', y = 'Height', hue = 'Sex', saturation = 1,   data = summer) 
plt.xticks(rotation = 45)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Height in cm', fontsize = 12)
plt.title('Height and Gender, by Year' , fontsize = 14)

# Weight and Gender by year
plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Year', y = 'Weight', hue = 'Sex', saturation = 1,   data = summer) 
plt.xticks(rotation = 45)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Weight by kg', fontsize = 12)
plt.title('Weight and Gender, by Year' , fontsize = 14)

# Age and Gender by year
plt.figure(figsize=(12, 8))
sns.boxplot(x = 'Year', y = 'Age', hue = 'Sex', saturation = 1,   data = summer) 
plt.xticks(rotation = 45)
plt.xlabel('Year', fontsize = 12)
plt.ylabel('Age', fontsize = 12)
plt.title('Age and Gender, by Year' , fontsize = 14)



###







