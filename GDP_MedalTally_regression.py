#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 18:20:07 2020

@author: Mansi Sethi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno  # pip install missingno if not installed already

pd.set_option('display.max_columns',None)
path='/Users/raghav/Desktop/documents/courses ba cu denver/bana 6620 ba computing/project_olympics/'


# merging NOC data that has country info with olympics data
NOC = pd.read_csv('{}/noc_regions.csv'.format(path))
OlympicsData = pd.read_csv('{}athlete_events.csv'.format(path))

OlympicsDat = OlympicsData.drop_duplicates()

print(OlympicsDat.isnull().sum()) # Printing columns with Null values

# looking at missing values visually:
missingno.matrix(OlympicsDat) # can be commented out if package missingno is missing

# Replacing blanks for medals with 'No Medal'
OlympicsDat['Medal'].fillna('No Medal',inplace=True ) 

OlympicsData.isnull().sum() #Looking at empty columns again
  
#keeping only needed columns .Dropping ID, Name, Games, city   
cats = ["Sex", "Age", "Height", "Weight", "Team", "NOC", "Year", "Season",
        "Sport", "Event", "Medal"]

OlympicsDat = OlympicsDat.loc[:,cats] 

# merging: left: use only keys from left frame, similar to a SQL left outer 
# join preserve key order.
# join on column NOC which is common to both the tables.
Olympic = pd.merge(OlympicsDat, NOC, how='left', on='NOC')
Olympic = Olympic.drop(['notes','NOC'], axis=1)
Olympic.head()

# read GDP data
GDPdata = pd.read_csv('{}GDP.csv'.format(path), skiprows=3)
GDPdata.rename(columns={'Country Name':'Country'},inplace=True)

#dropping extra columns
GDPdat = GDPdata.drop(['Country Code','Indicator Name','Indicator Code'], 
                      axis=1)

GDP_dat = pd.melt(GDPdat, 
            id_vars='Country', 
            value_vars=list(GDPdat.columns[1:]), 
            var_name='Year', 
            value_name='GDP')
GDP_dat.sort_values(['Country','Year'],ascending = [True,True],inplace=True)
GDP_dat.head()
GDP_dat.info() #year is not integer here
GDP_dat = GDP_dat.dropna()
GDP_dat = GDP_dat[~GDP_dat['Year'].str.contains('^Unnamed')]
GDP_dat['Year'] = GDP_dat['Year'].astype(int)

len(set(Olympic['region'])-set(GDP_dat.Country)) # number of country name mismatch

to_replace = ['Bahamas, The','Egypt, Arab Rep.','Iran, Islamic Rep.',
              "Cote d'Ivoire",'Kyrgyz Republic','North Macedonia',
              'Korea, Dem. People’s Rep.','Russian Federation',
              'Slovak Republic','Korea, Rep.','Syrian Arab Republic',
              'Trinidad and Tobago','United Kingdom','United States',
              'Venezuela, RB','Virgin Islands (U.S.)', 'Antigua and Barbuda',
              'Bolivia', 'Brunei Darussalam', 'Dem. Rep.,Gambia',
              'St. Kitts and Nevis','British Virgin Islands','Yemen, Rep.',
              'St. Vincent and the Grenadines','St. Lucia','Republic of Congo', 
              'The,Lao PDR','Democratic Republic of the Congo']
            

new_countries =   ['Bahamas', 'Egypt', 'Iran','Ivory Coast','Kyrgyzstan',
                   'Macedonia','North Korea','Russia','Slovakia','South Korea',
                   'Syria','Trinidad','UK','USA','Venezuela',
                   'Virgin Islands, US','Antigua','Boliva', 'Brunei','Gambia',
                   'Saint Kitts','Virgin Islands, British','Yemen',
                   'Saint Vincent','Saint Lucia','Fed. Sts.,Congo, Rep.', 
                   'The,Lao PDR','Congo']

#drop 'Cape Verde',Cook Islands,'Individual Olympic Athletes','Palestine','Taiwan',Micronesia

GDP_dat.replace(to_replace,new_countries,inplace=True)
mismatch_country_names = set(Olympic['region'])-set(GDP_dat.Country)
print("Number of country name mismatched after cleaning the data", 
      len(mismatch_country_names))

# drop where countries do not match
Olympic = Olympic[~Olympic['region'].isin(mismatch_country_names)] 
Olympic = Olympic[~Olympic['Medal'].isin(['No Medal'])]
GDP_dat_selected_range = GDP_dat[GDP_dat['Year'].isin(list(range(1980,2017)))]
oly_gdp_merged = Olympic.merge(GDP_dat_selected_range, 
                               left_on=['region','Year'],
                               right_on=['Country','Year'],
                               how='inner')
oly_gdp_merged_summer = oly_gdp_merged[oly_gdp_merged['Season']=='Summer']
medals_tally = oly_gdp_merged_summer.groupby(['Year','region']).size().\
    reset_index(name='counts')

medals_tally_top_5 = oly_gdp_merged.groupby(['region']).size().\
    reset_index(name='counts').sort_values('counts')[-5:]


# world map

#installation
pip install pycountry-convert
import pycountry
#function to convert to alpah2 country codes and continents
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2
def get_continent(col):
    if col=='UK':
        return ('GB','EU') # To convert values for UK
    try:
        cn_a2_code =  country_name_to_country_alpha2(col)
    except:
        cn_a2_code = 'Unknown' 
    try:
        cn_continent = country_alpha2_to_continent_code(cn_a2_code)
    except:
        cn_continent = 'Unknown' 
    return (cn_a2_code, cn_continent)


medals_tally['codes']=medals_tally['region'].apply(get_continent)

medals_tally[['Country', 'Continent']] = pd.DataFrame(medals_tally['codes'].tolist(), index=medals_tally.index) 

def get_country_alpha_code(col):
    try:
        cn_a3_code =  pycountry.countries.get(alpha_2=col).alpha_3
    except:
        cn_a3_code = 'Unknown' 
    return cn_a3_code

medals_tally['iso_alpha']=medals_tally['Country'].apply(get_country_alpha_code)
unknown_country = set(medals_tally[medals_tally['Country']=='Unknown']['region']) # {Trinidad, Kosovo}
medals_tally = medals_tally[~medals_tally['region'].isin(unknown_country)]


import plotly.express as px

fig = px.scatter_geo(medals_tally, locations="iso_alpha", color="Continent",
                     hover_name="region", size="counts",
                     animation_frame="Year",
                     projection="natural earth")
fig.update_layout(
    title="Country-wise distribution of medals (1980-2016)",
    title_x=0.5)

plot(fig)
#https://htmlpreview.github.io/?https://github.com/sethim20/olymics_analysis_project/blob/main/temp-plot.html



medals_tally.sort_values('counts',ascending=False)[:5]

final_oly_gdp_merged_medals = medals_tally.merge(GDP_dat_selected_range, 
                                                 left_on=['Year','region'],
                                                 right_on=['Year','Country'],
                                                 how='inner')

x = final_oly_gdp_merged_medals['GDP']
x_log = np.log2(x)
y = final_oly_gdp_merged_medals['counts']
y_log = np.log2(y)

df=pd.DataFrame({'GDP':x_log,'medal_tally_counts':y})
import plotly.express as px

fig = px.scatter(df, x="GDP", y="medal_tally_counts")
plot(fig)

##################### code starts
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import operator
from sklearn.preprocessing import PolynomialFeatures


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=40)
model.fit(np.array(X_train).reshape(-1,1), y_train)
y_pred = model.predict(np.array(X_test).reshape(-1,1))

plt.scatter(X_test,y_test )
plt.plot(X_test, y_pred, color='r')
plt.show()


rmse = np.sqrt(mean_squared_error(y_test,y_pred))
r2 = r2_score(y_test,y_pred)
print(rmse, r2)

x.corr(y)



sns_plot = sns.jointplot(x="GDP",y="counts", data=final_oly_gdp_merged_medals, kind="reg")
sns_plot.savefig("/Users/raghav/Desktop/documents/courses ba cu denver/bana 6620 ba computing/project_olympics/linear_regression_model.png")



g = sns.jointplot(x="Year",y="GDP", 
              data=GDP_dat_selected_range[GDP_dat_selected_range['Country']=='USA'], 
              kind="reg", joint_kws={'line_kws':{'color':'red'}})
g.savefig("/Users/raghav/Desktop/documents/courses ba cu denver/bana 6620 ba computing/project_olympics/linear_regression_model.png")



######################code ends

country_list =[]
predict_medal_counts=[]
predicted_gdp=[]
for country in set(oly_gdp_merged['Country']):
    country_list.append(country)
    country_data = GDP_dat_selected_range[GDP_dat_selected_range['Country']==country]
    country_model = LinearRegression()
    country_model.fit(np.array(country_data['Year']).reshape(-1, 1), country_data['GDP'])
    
    country_gdp_2020 = country_model.predict([[2020]])
    predicted_gdp.append(country_gdp_2020[0])
    predict_medal_counts.append(int(model.predict([country_gdp_2020])[0]))
    
final_predicted_df = pd.DataFrame({'predict_medal_count':predict_medal_counts,
                                   'region': country_list,
                                   'predicted_gdp_2020':predicted_gdp}).\
    sort_values('predict_medal_count')
oly_gdp_2016_df = final_oly_gdp_merged_medals[final_oly_gdp_merged_medals['Year']==2016]
#oly_gdp_2016_df = oly_gdp_2016_df.drop(['codes', 'Country_x','Continent','iso_alpha','Country_y'],axis=1)

final_df_2020_2016 = oly_gdp_2016_df.merge(final_predicted_df,on="region",how="left")
final_df_2020_2016.sort_values('predict_medal_count',ascending=False)[:5]
