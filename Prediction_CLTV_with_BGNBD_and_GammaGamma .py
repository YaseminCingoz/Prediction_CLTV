#!/usr/bin/env python
# coding: utf-8

# ## BUSINESS PROBLEM

# In[1]:


#The UK-based retail company wants to determine a roadmap for its sales and marketing activities. 
#In order for the company to make medium-long term plans, it is necessary to estimate the potential 
#value that existing customers will provide to the company in the future.


# In[2]:


pip install lifetimes


# ### DATASET

# In[35]:


import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


# In[4]:


# CLTV = Expected Number of Transaction * Expected Average Profit
# CLTV = (customer value / churn rate) * profit margin
# (Customer Value = Purcahse Frequency * Average Order Value)
# CLTV = BG/NBD Model * Gamma Gamma Submodel


# #### BG/NBD MODEL(Beta Geometric/Negative Binomial Distribution)

# In[5]:


# We will find expected number of transaction with BG/NBD Model
# BG/NBD(Beta Geometric/Negative Binomial Distribution)(BUY TILL YOU DIE)
# Modeling the distribution of the general population through probability and reduce it to the indivual

# BG/NBD Model models two processes for Expected Number of Transaction:
    # Transaction Process(Buy)  + Dropout Process(Till You Die)
    
    #Transaction Process: The number of transactions that can be performed by a customer in a certain period of 
    #time, as long as he is alive, is distributed with the transaction rate parameter. 
    ## That is, as long as a customer is alive, he will continue to make random purchases around his transaction rate
    # Transaction rates each. varies specific to the customer, gamma is distributed for the entire audience (r, a)
    #We can reduce the mass by inferring it from a population whose distribution you know.


# DROPUT PROCESS(TILL YOU DIE): each customer has a dropout rate with probability p
# Dropout rates vary for each customer and beta is distributed for the entire audience (a,b)


# #### GAMMA GAMMA MODEL

# In[6]:


# Used to estimate how much profit a customer can generate on average per transaction
# We will find expected average profit
# The monetary value of a customer's transactions is randomly distributed around the average of the transaction values.
# Average transaction value may vary between users over time, but does not vary for an individual user
# Expected transaction value is gamma distributed among all customers


# ## DATA PREPARATION

# In[7]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.4f' %x)


# In[8]:


# Define outlier_thresholds and replace_with_thresholds functions required to suppress outliers.
# When calculating cltv values, freq values must be int. Therefore, round the upper and lower values with round().

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

    
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


# In[9]:


df_ = pd.read_excel("/Users/yasemincingoz/Desktop/CLTV_Prediction/online_retail_II.xlsx")
df= df_.copy()
df.describe().T


# In[10]:


# There are minus values in Qunatity and price. We have to fix these.


# In[11]:


df.head()


# In[12]:


df.isnull().sum()


# In[13]:


df.dropna(inplace=True)
# Drop the null values. If we don't know the customer ID, we can't know  and predict who is the best customer or will be.


# In[14]:


df.describe().T
# after we drop the null values, price turned out to be normal but there is still problem in Quantity.


# In[15]:


# Retrieve the columns that not starts with 'C' in Invoice. Columns starts with 'C' shows the returned products.
df = df[~df['Invoice'].str.contains('C', na=False)]
df.describe().T

# The problem in Quantity columns is fixed.


# In[16]:


df = df[df['Quantity'] > 0]
df = df[df['Price'] > 0]
df.describe().T


# #### DATA ANALYSIS

# In[17]:


# We will calculate threshold values for Quantity. 
# Then it will replace the values above those threshold values with the threshold value.

replace_with_thresholds(df, 'Quantity')
replace_with_thresholds(df, 'Price')
df.describe().T


# In[18]:


df['TotalPrice'] = df['Quantity'] * df['Price']
df


# In[19]:


df["InvoiceDate"].max()


# In[20]:


# the day we made the analysis
today_date = dt.datetime(2010, 12, 11)


# ### PREPARATION OF LIFTIME DATA STRUCTURE

# In[21]:


#recency: time since last purchase
#T: age of the customer. Weekly (how long before the date of analysis was the first purchase made)
#Frequency: total number of recurring purchases (freq>1)
#monetary value: average earnings per purchase


# In[22]:


cltv_df = df.groupby('Customer ID').agg({'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                                                        lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
                                        'Invoice': lambda num: num.nunique(),
                                        'TotalPrice': lambda TotalPrice: TotalPrice.sum()})
cltv_df.head()


# In[23]:


cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head()


# In[24]:


# Monetary value: average earnings per purchase

cltv_df['monetary'] = cltv_df['monetary'] / cltv_df['frequency']
cltv_df.describe().T


# In[25]:


cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
cltv_df.describe().T


# In[26]:


#Convert recency and T to the weekly value

cltv_df['recency'] = cltv_df['recency'] / 7
cltv_df['T'] = cltv_df['T'] / 7
cltv_df.describe().T


# ## ESTABLISHMENT of BG-NBD MODEL

# In[27]:


bgf = BetaGeoFitter(penalizer_coef=0.001)
#This creates an instance of the BetaGeoFitter class. 
#The penalizer_coef parameter is used for regularization to prevent overfitting. 

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
#It fits the BG/NBD model to the data. 


# In[28]:


# Who are the 10 customers we expect to make the most purchases within 1 week?"

bgf.conditional_expected_number_of_purchases_up_to_time(1, 
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)

# We can also use bgf.predict()
# we found the customer ID's and expected number of purchases


# In[29]:


cltv_df['expected_purc_1_week'] = bgf.predict(1, 
                                            cltv_df['frequency'],
                                            cltv_df['recency'],
                                            cltv_df['T'])
cltv_df.head()


# In[30]:


# Who are the 10 customers we expect to make the most purchases within 1 month?

bgf.predict(4, 
         cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T']).sort_values(ascending=False).head(10)


# In[31]:


bgf.predict(4, 
         cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T']).sum()


# In[32]:


# Who are the 10 customers we expect to make the most purchases within 12 months?"

cltv_df["expected_purch_12_month"] = bgf.predict(4 * 12, 
         cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])


# In[33]:


bgf.predict(4 * 12, 
         cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T']).sort_values(ascending=False).head(10)


# In[36]:


# Evaluation of predicted results

import matplotlib.pyplot as plt
plot_period_transactions(bgf)
plt.show()


# ## ESTABLISHMENT of GAMMA-GAMMA MODEL

# In[38]:


ggf = GammaGammaFitter(penalizer_coef = 0.01)
ggf.fit(cltv_df['frequency'],
        cltv_df['monetary'])


# In[39]:


# Conditional expected avg profit values

ggf.conditional_expected_average_profit(
        cltv_df['frequency'],
        cltv_df['monetary']).sort_values(ascending=False).head(10)


# In[40]:


cltv_df['expected_average_profit'] = ggf.conditional_expected_average_profit(
        cltv_df['frequency'],
        cltv_df['monetary'])


# In[41]:


cltv_df.sort_values('expected_average_profit', ascending=False).head(10)


# In[ ]:


############################
# CALCULATION OF CLTV With BG/NBD ANF GG MODEL


# In[42]:


cltv = ggf.customer_lifetime_value(bgf,
                                cltv_df['frequency'],
                                cltv_df['recency'],
                                cltv_df['T'],
                                cltv_df['monetary'],
                                time=6, #6 aylik
                                freq = 'W', # weekley
                                discount_rate = 0.01)
cltv.head()


# In[43]:


cltv = cltv.reset_index()


# In[44]:


cltv_final = cltv_df.merge(cltv, on='Customer ID', how='left')
cltv_final.sort_values(by='clv', ascending=False).head(10)


# In[45]:


# CREATING THE CUSTOMER SEGMENT


# In[46]:


cltv_final['segment'] = pd.qcut(cltv_final['clv'], 4, labels=['D','C','B','A'])
cltv_final.head(20)


# In[47]:


cltv_final.groupby('segment').agg({'count', 'mean', 'sum'})


# In[ ]:




