#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#https://www.kaggle.com/shinjinibasu/summer-olympics-exploration-and-linear-regression
#https://www.kaggle.com/mytymohan/120-years-of-olympic-data-analysis


#https://plotly.com/python/figure-labels/
#https://plotly.com/python/line-charts/#sparklines-with-plotly-express
'''It could also be interesting to see something like an exploratory data analysis over time. 
For example, what is the medal distribution (by country or continent) in the first 40 years of the modern Olympics? 
How about in years 40-80 ? Or years 80-120? Or if you want to take it a step further 
and animate the changes over time, look into the imageio package.
'''

"""FEEDBACK AND COMMENTS:
Suppose you plotted a bar chart for each olympic year and used plt.savefig().

e.g.
for olympicYear in olympicYearList:
#some plt commands that plot medals by continent, for example
filename = str(olympicYear)+'.jpg'
plt.savefig(filename)
filenameList.append(filename)


import imageio
images = []
for filename in fileNameList:
images.append(imageio.imread(filename))
imageio.mimsave('myAnimation.gif', images)


If you can find a nice dataset with historic economic and demographic information for each country
 (e.g., from the world bank or IMF), then it could be interesting to cross-reference 
 the medal outcome vs. the economic or population situation for various countries.
""
Created on Sat Oct 17 23:45:39 2020

@author: Mansi Sethi
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


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
#pip install missingno
import missingno
missingno.matrix(OlympicsDat)
 

#REPLACING BLANKS FOR MEDALS WITH 'No Medal'
OlympicsDat['Medal'].fillna('No Medal',inplace=True ) 

#Looking at empty columns again:
OlympicsData.isnull().sum()
  
#keeping only needed columns 
#dropping ID, Name, Games, city   
cats = ["Sex","Age","Height","Weight","Team","NOC","Year","Season","Sport","Event","Medal"]

OlympicsDat = OlympicsDat.loc[:,cats] 

OlympicsDat.head(5)
OlympicsDat.describe() 

#no of unique events
Count_events=OlympicsDat.groupby(by=['Year','Season'],as_index=False)['Event'].count()

#women in olympics vs men
Year_Sex_count=OlympicsDat.groupby(by=['Year','Sex'])['Event'].\
    count().reset_index(name="Number of participants")


#creating data table for males and females separately
male_partipant_count = Year_Sex_count.loc[Year_Sex_count["Sex"]=="M"]

female_partipant_count =Year_Sex_count.loc[Year_Sex_count["Sex"]=="F"]

#merging male_participant_count with Year_sex_season_count 
#replacing NaNs with zero in case where males did not participate
male_partipant_count = pd.DataFrame(Year_Sex_count['Year']).\
    merge(male_partipant_count, on='Year', how='left').fillna(0)

#merging female_participant_count with Year_sex_season_count 
#this is to get zeroes for years when female did not get to participate
#used left to merge to get all the years from Year_Sex_count 
female_partipant_count = pd.DataFrame(Year_Sex_count['Year']).\
    merge(female_partipant_count, on='Year', how='left').fillna(0)

''''overall graph with both summer and winter
#plotting using plotly
import plotly.graph_objects as go
from plotly.offline import plot
x = Year_Sex_count['Year']
y0 = male_partipant_count['Number of participants']
y1 = female_partipant_count['Number of participants']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='male'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='female'))
plot(fig)
'''''
#data based on season, sex and year aggregation and naming count as No. of participant:
#women in olympics vs men
Year_Sex_season_count=OlympicsDat.groupby(by=['Year','Sex','Season'])['Event'].\
    count().reset_index(name="Number of participants")

#grouping based on year, gendere, season and country
Year_Sex_season_country_count=OlympicsDat.groupby(by=['Year','Sex','Season','NOC'])['Event'].\
    count().reset_index(name="Number of participants")
#  Male participants US
male_partipant_ct = Year_Sex_season_country_count.loc[Year_Sex_season_country_count["Sex"]=="M"]
male_US_ct=male_partipant_ct.loc[male_partipant_ct["NOC"]=="USA"]
male_US_winter = male_US_ct.loc[male_US_ct["Season"]=="Winter"]
male_US_summer = male_US_ct.loc[male_US_ct["Season"]=="Summer"]

# FEmale participants US
female_partipant_ct = Year_Sex_season_country_count.loc[Year_Sex_season_country_count["Sex"]=="F"]
female_US_ct=female_partipant_ct.loc[female_partipant_ct["NOC"]=="USA"]
female_US_winter = female_US_ct.loc[female_US_ct["Season"]=="Winter"]
female_US_summer = female_US_ct.loc[female_US_ct["Season"]=="Summer"]

###female got to participated later. so, merging the data on year basis with male and 
#adding zero for those non participation
female_US_summer = pd.DataFrame(male_US_summer['Year']).\
    merge(female_US_summer, on='Year', how='outer').fillna(0)
    
#plotting using plotly 
import plotly.graph_objects as go
from plotly.offline import plot
x = male_US_summer['Year']
y0 = male_US_summer['Number of participants']
y1 = female_US_summer['Number of participants']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Male'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Female'))
fig.update_layout(
    title="Male vs Female participation (US) in Summer Olympics",
    xaxis_title="Year",
    yaxis_title="No. of US participants")
plot(fig)




import plotly.graph_objects as go
from plotly.offline import plot
x = male_US_winter['Year']
y0 = male_US_winter['Number of participants']
y1 = female_US_winter['Number of participants']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Male'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Female'))
fig.update_layout(
    title="Male vs Female participation (US) in Winter Olympics",
    xaxis_title="Year",
    yaxis_title="No. of US participants")
plot(fig)

#winter olympic
import plotly.graph_objects as go
from plotly.offline import plot
x = male_partipant_winter['Year']
y0 = male_partipant_winter['Number of participants']
y1 = female_partipant_winter['Number of participants']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Male'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Female'))
fig.update_layout(
    title="Male vs Female participation in Winter Olympics",
    xaxis_title="Year",
    yaxis_title="No. of participants")
plot(fig)

############################################


############overall male vs female comparison###########
#creating data table for males and females separately. Then segregating
#each of them based on seasons separately in preparation of graphs
male_partipant_count = Year_Sex_season_count.loc[Year_Sex_season_count["Sex"]=="M"]
male_partipant_summer = male_partipant_count.loc[male_partipant_count["Season"]=="Summer"]
male_partipant_winter = male_partipant_count.loc[male_partipant_count["Season"]=="Winter"]

female_partipant_count =Year_Sex_season_count.loc[Year_Sex_season_count["Sex"]=="F"]
female_partipant_summer = female_partipant_count.loc[female_partipant_count["Season"]=="Summer"]
female_partipant_winter = female_partipant_count.loc[female_partipant_count["Season"]=="Winter"]


#merging female_participant_count with Year_sex_season_count 
#this is to get zeroes for years when female did not get to participate
#used left to merge to get all the years from Year_Sex_count 
female_partipant_summer = pd.DataFrame(male_partipant_summer['Year']).\
    merge(female_partipant_summer, on='Year', how='outer').fillna(0)
    
    
#merging male_participant_count with Year_sex_season_count 
#replacing NaNs with zero in case where males did not participate
female_partipant_winter = pd.DataFrame(male_partipant_winter['Year']).\
    merge(female_partipant_winter, on='Year', how='outer').fillna(0)


#plotting using plotly
import plotly.graph_objects as go
from plotly.offline import plot
x = male_partipant_summer['Year']
y0 = male_partipant_summer['Number of participants']
y1 = female_partipant_summer['Number of participants']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Male'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Female'))
fig.update_layout(
    title="Male vs Female participation in Summer Olympics",
    xaxis_title="Year",
    yaxis_title="No. of participants")
plot(fig)


#winter olympic
import plotly.graph_objects as go
from plotly.offline import plot
x = male_partipant_winter['Year']
y0 = male_partipant_winter['Number of participants']
y1 = female_partipant_winter['Number of participants']


# Create traces
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y0,
                    mode='lines+markers',
                    name='Male'))
fig.add_trace(go.Scatter(x=x, y=y1,
                    mode='lines+markers',
                    name='Female'))
fig.update_layout(
    title="Male vs Female participation in Winter Olympics",
    xaxis_title="Year",
    yaxis_title="No. of participants")
plot(fig)

'''merging NOC data that has country info with olympics data'''
NOC = pd.read_csv('/Users/raghav/Desktop/documents/courses ba cu denver/bana 6620 ba computing/project_olympics/noc_regions.csv')
#merging: left: use only keys from left frame, similar to a SQL left outer join; preserve key order.
#join on column NOC which is common to both the tables.
Olympic = pd.merge(OlympicsDat,NOC,how='left',on='NOC')
for colName in Olympic:
    print(colName)
Olympic=Olympic.drop(['notes','NOC'],axis=1)
Olympic.head()

'''''''
#GDP

''''''''
GDPdata.rename(columns={'Country Name':'Country'},inplace=True)
#dropping extra columns
GDPdat=GDPdata.drop(['Country Code','Indicator Name','Indicator Code'],axis=1)

#print out all column names
for colName in GDPdat:
    print(colName)

'''using pandas.melt id_vars: columns to use as identifier variable,
value_vars:: columns to unpivot, var_name: name to use the unpivoted variable, 
value_name
'''

GDP_dat = pd.melt(GDPdat, 
            id_vars='Country', 
            value_vars=list(GDPdat.columns[1:]), 
            var_name='Year', 
            value_name='GDP')
GDP_dat.sort_values(['Country','Year'],ascending = [True,True],inplace=True)
GDP_dat.head()
GDP_dat.info() #year is not integer here

GDP_dat = GDP_dat[~GDP_dat['Year'].str.contains('^Unnamed')]
GDP_dat(GDP_dat.columns[GDP_dat.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
GDP_dat.dropna(how='all', axis='columns')

GDP_dat['Year'] = GDP_dat['Year'].astype(int)
#GDP_dat[(GDP_dat.Year >= '1980') & (GDP_dat <= '2016')]


GDP_dat[GDP_dat.some_date.between(1980, 2016)]
GDP_dat['Year'] = GDP_dat['Year'].astype(int)
GDP_dat=GDP_dat.drop(labels=16367)

#finding countries that don't match between olympics data and GDP data by using set and difference
Olympic.where(OlympicsDat.Team==GDP_dat.Country)
set(set(GDP_dat.Country)-set(Olympic['region'])
|
    len(set(Olympic['region'])-set(GDP_dat.Country))
#relacing country names in GDP with relevant ones in Olympics to facilitate match and merging afterwards
to_replace = ['Bahamas, The','Egypt, Arab Rep.','Iran, Islamic Rep.',"Cote d'Ivoire",'Kyrgyz Republic',
              'North Macedonia',
             'Korea, Dem. People’s Rep.','Russian Federation','Slovak Republic','Korea, Rep.','Syrian Arab Republic',
              'Trinidad and Tobago','United Kingdom','United States','Venezuela, RB','Virgin Islands (U.S.)', 'Antigua and Barbuda','Bolivia',
              'Brunei Darussalam', 'Dem. Rep.,Gambia' ,
              'St. Kitts and Nevis','British Virgin Islands','Yemen, Rep.','St. Vincent and the Grenadines','St. Lucia','Republic of Congo', 
              'The,Lao PDR','Democratic Republic of the Congo']
            

new_countries =   ['Bahamas',
                   'Egypt', 'Iran', 
                   'Ivory Coast','Kyrgyzstan',
                   'Macedonia','North Korea','Russia','Slovakia',
                   'South Korea','Syria','Trinidad','UK',
                   'USA','Venezuela','Virgin Islands, US',
                   'Antigua','Boliva', 'Brunei',
                   'Gambia',
                   'Saint Kitts',
                   'Virgin Islands, British','Yemen',
                   'Saint Vincent','Saint Lucia',
                   'Fed. Sts.,Congo, Rep.', 
                   'The,Lao PDR','Congo']

#drop 'Cape Verde',Cook Islands,'Individual Olympic Athletes','Palestine','Taiwan',Micronesia

GDP_dat.replace(to_replace,new_countries,inplace=True)
Olympic.shape
Olympic = Olympic[~Olympic['region'].isin(set(Olympic['region'])-set(GDP_dat.Country))] # drop where countries do not match
Olympic.shape
Olympic.columns


GDP_dat_20_years = GDP_dat[GDP_dat['Year'].isin(list(range(1980,2017)))]

final_oly_gdp_merged = Olympic.\
    merge(GDP_dat_20_years, left_on='region',right_on='Country', how='inner')

sum(GDP_dat_20_years['GDP']==np.NaN)

sum(final_oly_gdp_merged['GDP']==np.nan)
sum(final_oly_gdp_merged['Year_x']==0)
final_oly_gdp_merged.head()

final_oly_gdp_merged.shape

'''medals tally'''
medals_tallyT = final_oly_gdp_merged.groupby(['Year_x', 'Country']).size().reset_index(name='counts')
medals_tallyT.head()

final_oly_gdp_merged_medals = medals_tally.\
    merge(GDP_dat_20_years, left_on=['Year_x','Country'],right_on=['Year','Country'], how='inner')

final_oly_gdp_merged_medals = final_oly_gdp_merged_medals[final_oly_gdp_merged_medals['counts']>=10]

final_oly_gdp_merged_medals = final_oly_gdp_merged_medals.dropna(subset=['GDP'])

plt.scatter(np.log2(final_oly_gdp_merged_medals['GDP']), 
            (final_oly_gdp_merged_medals['counts']))

float(np.correlate(np.log2(final_oly_gdp_merged_medals['GDP']),final_oly_gdp_merged_medals['counts'] )[0])

correlation = final_oly_gdp_merged_medals['GDP'].corr(final_oly_gdp_merged_medals['counts'])

from sklearn.linear_model import LinearRegression
x_log = np.log2(final_oly_gdp_merged_medals['GDP'])[:, np.newaxis]
x=final_oly_gdp_merged_medals['GDP'][:, np.newaxis]
y_log = np.log2(final_oly_gdp_merged_medals['counts'])[:, np.newaxis]
y=final_oly_gdp_merged_medals['counts'][:, np.newaxis]

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=40)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test,y_test )
plt.plot(X_test, y_pred, color='r')
plt.show()

plt.scatter(x,y)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse, r2)


# with log value on y-axis
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(x_log, y, test_size=0.33, random_state=40)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test,y_test )
plt.plot(X_test, y_pred, color='r')
plt.show()

plt.scatter(x_log,y)
rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse, r2)


# both values in log
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(x_log, y_log, test_size=0.33, random_state=40)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.scatter(X_test,y_test )
plt.plot(X_test, y_pred, color='r')
plt.show()

rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse, r2)

import operator

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures


# transforming the data to include another axis

polynomial_features= PolynomialFeatures(degree=2)
x_poly = polynomial_features.fit_transform(x_log)

model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
r2 = r2_score(y,y_poly_pred)
print(rmse)
print(r2)

plt.scatter(x_log, y, s=10)
# sort the values of x before line plot
sort_axis = operator.itemgetter(0)
sorted_zip = sorted(zip(x_log,y_poly_pred), key=sort_axis)
x, y_poly_pred = zip(*sorted_zip)
plt.plot(x_log, y, color='m')
plt.show()

sns.jointplot(x, y, data=nbadata, kind="reg")




medals_tally['Medal_Count'] = medals_tally['Medal_Won']/(medals_tally['Team_Event']+medals_tally['Individual_Event'])


pip install statsmodels

import statsmodels.api as sm

cats = ["Sex","Age","Height","Weight","Team","NOC","Year","Season","Sport","Event","Medal"]

OlympicsDat = OlympicsDat.loc[:,cats] 

final_oly_gdp_merged_medals
data=final_oly_gdp_merged_medalscount_10

for col in oly_gdp_merged: 
    print(col) 

cats = ["Age","Height","Weight","Team"]
oly_gdp_corr= oly_gdp_merged.loc[:,cats] 
    
import seaborn as sn
corrMatrix = oly_gdp_corr.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


for col in final_oly_gdp_merged_medals: 
    print(col) 
    

corrMatrix = final_oly_gdp_merged_medals.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()

#no of medals, no of participants and gdp: CORRELATION MATRIX
#regression
#world map with no of medals DONE


#show the map
world_map

import plotly.express as px
px.data.gapminder().query("year==2007")
