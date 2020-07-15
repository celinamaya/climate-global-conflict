#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


#read in files
#change to relative paths rather than absolute paths
aid_df = pd.read_csv('/home/ubuntu/us_foreign_aid_complete.csv')
deaths_df = pd.read_csv('/home/ubuntu/Violent Death Per Hundred-Thousand.csv')

deaths_df.head()


# In[3]:


#Switch Year row-columns 
deaths_df = deaths_df.melt('Country', var_name='Year', value_name='Deaths')
deaths_df.head(100)
#len(deaths_df['Country'])


# In[4]:


aid_df.head()


# In[5]:


aid_df.loc[aid_df.fiscal_year == '1976tq'] = 1976

aid_df['Fiscal Year Clean'] = pd.to_numeric(aid_df['fiscal_year'], errors='coerce')

#aid_df['Fiscal Year Clean'].unique()


# In[6]:


#get rid of nans
aid_df.dropna(how='all')

#select data for years 2004 onward
cln_aid_df = aid_df[aid_df['Fiscal Year Clean'] >=2004]
print(len(cln_aid_df))
#Get rid of all regional groupings in the country_name column
cln_aid_df = cln_aid_df[cln_aid_df['country_name'] != 'World']
cln_aid_df = cln_aid_df[cln_aid_df['country_name'].str.contains('Region') == False]
print(len(cln_aid_df))


# In[7]:


country_diffs = list(set(cln_aid_df['country_name']) - set(deaths_df['Country']))
country_diffs2 = list(set(deaths_df['Country']) - set(cln_aid_df['country_name']))

full_diffs = list(country_diffs + country_diffs2)


# In[8]:


deaths_df['Country'].unique()


# In[9]:


#cln_aid_df['country_name'].unique()


# In[10]:


full_diffs.sort()
print(full_diffs)


# In[11]:


new_names = ['', 'Antigua and Barbuda', 'Antigua and Barbuda', '', '', 'Brunei', 'Brunei', 'Burma', 'Burma', 
             'Cape Verde', 'Cape Verde', 'China', 'Hong Kong', 'China', 'China', 'Macao', 
             'Republic of Congo', 'Republic of Congo', 'Democratic Republic of Congo', 'Curacao', 'Curacao', 
             'Czechia', 'Czechia', 'Democratic Republic of Congo', 'Eswatini', 'Denmark', 'France', 'United States', 
             'Hong Kong','Iran', 'Iran', 'Korea Republic', 'Korea, Democratic Republic', 'Korea, Democratic Republic', 'Korea Republic', 
             "Laos", 'Laos', 'Libya', 'Libya', '', 'Macao', 'France', 'Micronesia (Federated States)', 
             'Micronesia (Federated States)', 'Moldova', '', '', '', 'North Macedonia', '', 'Palestine', 
             'United States', 'Moldova','France', 'St. Kitts and Nevis', 'St. Lucia', 'France', 
             'St. Vincent and Grenadines', '','Serbia', 'Slovakia', 'Slovakia', 'St. Kitts and Nevis', 'St. Lucia', 
             'St. Vincent and Grenadines','Sudan', 'Eswatini', 'Syria', 'Syria', 'Tanzania', 'North Macedonia', 
             'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'United Kingdom', 'Tanzania', 
             'United States','Palestine', '']


# In[12]:


new_name_dict = dict(zip(full_diffs, new_names))
print(new_name_dict)


# In[13]:


#df['col1'].map(di).fillna(df['col1'])

cln_aid_df['country_name'] = cln_aid_df['country_name'].map(new_name_dict).fillna(cln_aid_df['country_name'])

deaths_df['Country'] = deaths_df['Country'].map(new_name_dict).fillna(deaths_df['Country'])


# In[14]:


'''for item in new_name_dict.keys():
    if item in cln_aid_df['country_name']:
        cln_aid_df['country_name'].replace(item, new_name_dict[item], inplace=True)

#cln_aid_df['country_name'].unique()

for item in new_name_dict.keys():
    if item in deaths_df['Country']:
        deaths_df['Country'].replace(item, new_name_dict[item], inplace=True)'''


# In[15]:


#Use these to check for differences in country naming schemes along the way
cln_aid_df['country_name'].unique()


# In[16]:


deaths_df['Country'].unique()


# In[17]:


#Remove rows with empty '' in the country_name and Country columns
cln_aid_df = cln_aid_df[cln_aid_df['country_name'] != '']


# In[18]:


#Collapse all multiples, i.e. China2004's into a single line
aggregation_functions = {'current_amount': 'sum', 'constant_amount': 'sum', 'country_name': 'first', 
                         'Fiscal Year Clean':'first'}
grpd_cln_aid_df = cln_aid_df.groupby(cln_aid_df['country_name']).aggregate(aggregation_functions)

grpd_cln_aid_df.head(100)


# In[19]:


grpd_cln_aid_df['Violent Deaths'] = np.zeros(len(grpd_cln_aid_df['country_name']))

grpd_cln_aid_df.head(100)


# In[20]:


grpd_cln_aid_df['Country+Year'] = grpd_cln_aid_df['country_name'].map(str) + grpd_cln_aid_df['Fiscal Year Clean'].map(
    str)


# In[21]:


grpd_cln_aid_df.head()


# In[22]:


deaths_df['Country+Year'] = deaths_df['Country'].map(str) + deaths_df['Year'].map(str)
deaths_df.head()


# In[23]:


deaths_by_countryyear = dict(zip(deaths_df['Country+Year'], deaths_df['Deaths']))
print(deaths_by_countryyear)


# In[24]:


grpd_cln_aid_df['Violent Deaths'] = grpd_cln_aid_df['Country+Year'].map(deaths_by_countryyear)


# In[25]:


grpd_cln_aid_df.dropna(subset=['Violent Deaths'], inplace=True)


# In[26]:


grpd_cln_aid_df.head(100)


# In[27]:


sns.pairplot(grpd_cln_aid_df)


# In[28]:


grpd_cln_aid_df.corr()


# In[29]:


X = grpd_cln_aid_df[['Fiscal Year Clean', 'Violent Deaths']]
y = grpd_cln_aid_df['constant_amount']


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[31]:


lm = LinearRegression()
lm.fit(X_train,y_train)


# In[32]:


predictions = lm.predict(X_test)


# In[33]:


plt.scatter(y_test,predictions)


# In[34]:


cln_aid_df.head()


# In[35]:


cln_aid_df['Violent Deaths'] = np.zeros(len(cln_aid_df['country_name']))


# In[36]:


cln_aid_df['Country+Year'] = cln_aid_df['country_name'].map(str) + cln_aid_df['Fiscal Year Clean'].map(str)


# In[37]:


cln_aid_df['Violent Deaths'] = cln_aid_df['Country+Year'].map(deaths_by_countryyear)


# In[38]:


cln_aid_df.head(100)


# In[39]:


sub_cln_aid_df = cln_aid_df[['country_id', 'region_id', 
                'channel_subcategory_id', 
                'funding_agency_id','assistance_category_id','activity_id','transaction_type_id', 
                'fiscal_year','USG_sector_id','Violent Deaths', 'constant_amount']]


# In[40]:


sub2_cln_aid_df = sub_cln_aid_df[sub_cln_aid_df.columns].astype(float)
sub2_cln_aid_df.head()


# In[41]:


cor_tab = pd.DataFrame(sub2_cln_aid_df.corr())


# In[42]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# In[43]:


#np.isnan(sub2_cln_aid_df)
#sub2_cln_aid_df = sub2_cln_aid_df[sub2_cln_aid_df['Violent Deaths'] != np.nan]
sub2_cln_aid_df = sub2_cln_aid_df[sub2_cln_aid_df['fiscal_year'] <= 2015]
#sub2_cln_aid_df.interpolate()
sub2_cln_aid_df.fillna(0, inplace = True)


# In[44]:


#np.isnan(sub2_cln_aid_df)
#sub2_cln_aid_df.replace([np.inf, -np.inf], np.nan)
#sub2_cln_aid_df.loc[(~np.isfinite(sub2_cln_aid_df)) & sub2_cln_aid_df.notnull()] = np.nan
sub2_cln_aid_df.replace([np.inf, -np.inf], np.nan)
sub2_cln_aid_df.dropna(how="all")


# In[45]:


sub2_cln_aid_df.plot(kind='box', subplots=True, layout=(5,4), sharex=False, sharey=False)
plt.tight_layout()
plt.show()


# In[46]:


#Split out values for validation group
array = sub2_cln_aid_df.values
X = array[:,0:10]
Y = array[:,10]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)


# In[47]:


# Test options and evaluation metric
seed = 7
scoring = 'accuracy'


# In[48]:


#sub2_cln_aid_df =sub2_cln_aid_df[~sub2_cln_aid_df.isin([np.nan, np.inf, -np.inf]).any(1)]


# In[49]:


#REVERT TO FULL DATASET 


# In[50]:


np.where(np.isnan(sub2_cln_aid_df))
np.nan_to_num(sub2_cln_aid_df)
sub2_cln_aid_df.apply(lambda s: s[np.isfinite(s)].dropna())


# In[51]:


np.where(np.isnan(sub2_cln_aid_df))


# In[52]:


sub2_cln_aid_df.iloc[920,:16]


# In[53]:


np.any(np.isnan(sub2_cln_aid_df))


# In[54]:


np.all(np.isfinite(sub2_cln_aid_df))


# In[ ]:


models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[ ]:


'''Results of the last cell:
LR: 0.972418 (0.000602)
LDA: 0.970971 (0.000660)
KNN: 0.992250 (0.000398)
CART: 0.992856 (0.000323)
NB: 0.968353 (0.000608)'''


# In[ ]:


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:




