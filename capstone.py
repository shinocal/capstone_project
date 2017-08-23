
# coding: utf-8

# In[1]:


import ujson as json
import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
get_ipython().magic(u'load_ext sql')


# In[2]:


xl = pd.ExcelFile('zip_code_database.xls')
x = xl.sheet_names
sheet1=x[0]
zip_ref = xl.parse(sheet1)


# In[3]:


fda_xl = pd.ExcelFile('DataDownload.xls')
#fda_sheet_names=fda_xl.sheet_names
health_data = fda_xl.parse('HEALTH')
supp_health_data = fda_xl.parse('Supplemental Data - County')
food_fda = fda_xl.parse('RESTAURANTS')


# In[4]:


print health_data['FIPS'][health_data['FIPS']==2230].index


# In[5]:


print len(health_data)


# In[6]:


health_data=health_data.drop(health_data['FIPS'][health_data['FIPS']==2230].index)
health_data=health_data.drop(health_data['FIPS'][health_data['FIPS']==2275].index)
health_data=health_data.drop(health_data['FIPS'][health_data['FIPS']==2198].index)


# In[7]:


food_fda=food_fda.drop(food_fda['FIPS'][food_fda['FIPS']==2230].index)
food_fda=food_fda.drop(food_fda['FIPS'][food_fda['FIPS']==2275].index)
food_fda=food_fda.drop(food_fda['FIPS'][food_fda['FIPS']==2198].index)


# In[8]:


supp_health_data=supp_health_data.drop(supp_health_data['FIPS Code'][supp_health_data['FIPS Code']==2230].index)
supp_health_data=supp_health_data.drop(supp_health_data['FIPS Code'][supp_health_data['FIPS Code']==2275].index)
supp_health_data=supp_health_data.drop(supp_health_data['FIPS Code'][supp_health_data['FIPS Code']==2198].index)


# In[9]:


print len(health_data),len(food_fda),len(supp_health_data)


# In[10]:


nanind=[]
for i in range(0,len(health_data)):
    if np.isnan(health_data['PCT_DIABETES_ADULTS10'][i]):
        nanind.append(i)


# In[11]:


health_data=health_data.drop(nanind)
food_fda=food_fda.drop(nanind)
supp_health_data=supp_health_data.drop(nanind)


# In[12]:


print len(health_data),len(food_fda),len(supp_health_data)


# In[13]:


food_fda=food_fda.rename(columns = {'FIPS':'FIPS1'})
food_fda=food_fda.rename(columns = {'State':'State1'})
food_fda=food_fda.rename(columns = {'County':'County1'})


# In[164]:


print supp_health_data.head()


# In[15]:


#%%sql sqlite://
get_ipython().magic(u'sql DROP TABLE IF EXISTS food_fda')
get_ipython().magic(u'sql PERSIST food_fda')


# In[16]:


get_ipython().magic(u'sql DROP TABLE IF EXISTS health_data')
get_ipython().magic(u'sql PERSIST health_data')


# In[17]:


get_ipython().run_cell_magic(u'sql', u'', u'SELECT State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM health_data JOIN food_fda ON (FIPS=FIPS1) LIMIT 5')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[18]:


zip_dict=dict()
#for i in range(0,len(zip_ref['county'])):
#    zip_dict[str(zip_ref['zip'][i])]=zip_ref['county'][i]
zip_dict['county']=zip_ref['county']
zip_dict['zip']=zip_ref['zip']
zip_dict['state']=zip_ref['state']
zip_df=pd.DataFrame(zip_dict)


# In[19]:


county=[]
county=zip_df['county']
null_val=county.isnull()
nullcount=0
nullind=[]
t=0
for i in null_val:    
    if i: 
        nullcount+=1
        nullind.append(t)        
    t+=1            
print nullcount
print len(nullind)
zip_df=zip_df.drop(zip_df.index[[nullind]])


# In[20]:


drop_ind=[]
for index, row in zip_df.iterrows():
    if zip_df['county'][index][-6:]!='County':
        drop_ind.append(index)
zip_df=zip_df.drop(drop_ind)


# In[21]:


print zip_df['county'][0][:-7]


# In[22]:


county_list=[]
for index, row in zip_df.iterrows():
        county_list.append(zip_df['county'][index][:-7])


# In[23]:


zip_df['county']=county_list


# In[24]:


#for index, row in zip_df.iterrows():
#    zip_df['county'][index]=zip_df['county'][index][:-7]


# In[25]:


zip_df=zip_df.rename(columns = {'state':'state2'})
zip_df=zip_df.rename(columns = {'county':'county2'})


# In[26]:


get_ipython().magic(u'sql DROP TABLE IF EXISTS zip_df')
get_ipython().magic(u'sql PERSIST zip_df')


# In[27]:


get_ipython().magic(u'sql select * from zip_df limit 10')


# In[28]:


get_ipython().run_cell_magic(u'sql', u'', u'SELECT zip,State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM \nzip_df JOIN (SELECT State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM health_data JOIN food_fda ON (FIPS=FIPS1))\nON (county2=County and state2=State) LIMIT 5')


# In[166]:


X=np.column_stack((np.array(health_data['PCT_DIABETES_ADULTS10']),np.array(food_fda['FFRPTH12'])))


# In[ ]:


from sklearn.covariance import EllipticEnvelope


# In[181]:




EE=EllipticEnvelope(contamination=0.1)
EE.fit(X)
cs=EE.predict(X)



# In[196]:


ones=np.ones((len(health_data[cs==1])))
predictor=np.array([ones,list(food_fda['FFRPTH12'][cs==1])])
print len(list(health_data['PCT_DIABETES_ADULTS10'][cs==1])),predictor.shape
glm_model=sm.GLM(np.array(list(health_data['PCT_DIABETES_ADULTS10'][cs==1])),np.transpose(predictor))
glm_results=glm_model.fit()
print glm_results.params,glm_results.pvalues


# In[248]:


plt.plot(np.array(food_fda['FFRPTH12']),np.array(health_data['PCT_DIABETES_ADULTS10']), 'k.')
plt.plot(np.array(food_fda['FFRPTH12'])[cs==-1],np.array(health_data['PCT_DIABETES_ADULTS10'])[cs==-1], 'r.',)
#plt.plot(np.array(food_fda['FFRPTH12'])[cs==1],glm_results.mu, 'b-',)
#sns.regplot(x="FFRPTH12", y="diab", data=plot_df)
plt.ylim([0,30])
plt.xlim([0,2])
plt.xlabel('Fast Food / 1000 pop.')
plt.ylabel('% diabetic')
plt.title('FF density vs. Diabetes Rate')
#plt.show()
plt.savefig('fig1.png')


# In[214]:


plot_dict=dict()
plot_dict['FFRPTH12']=list(food_fda['FFRPTH12'][cs==1])
plot_dict['diab']=list(health_data['PCT_DIABETES_ADULTS10'][cs==1])
plot_df=pd.DataFrame(plot_dict)


# In[251]:


import seaborn as sns
#sns.regplot(x="FFRPTH12", y="diab", data=plot_df,joint_kws={'line_kws':{'color':'red'}})
sns.jointplot(x="FFRPTH12", y="diab", data=plot_df, kind='reg',joint_kws={'line_kws':{'color':'red'}})
#tips = sns.load_dataset("tips")
plt.xlim([0,1.5])
plt.ylim([0,30])
#plt.show()
plt.savefig('fig2.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[455]:


with open('yelp_academic_dataset_business.json') as f:
    data=[json.loads(line) for line in f]


# In[456]:


star_ratings=[row['stars'] for row in data]
categories=[row['categories'] for row in data]
yelpstates=[row['state'] for row in data]


# In[457]:


print len(star_ratings),len(categories)


# In[458]:


fastfood=[]
#fastfood.append('Restaurants')
fastfood.append('Fast Food')
#fastfood.append('Pizza')
#fastfood.append('Burgers')
#fastfood.append('Pretzels')
#fastfood.append('Chicken Wings')
#fastfood.append('Chicken Shop')
#fastfood.append('Comfort Food')
#fastfood.append('Delis') # added 8-17-17
#fastfood.append('Diners')
#fastfood.append('Donuts')
#fastfood.append('Fish & Chips')
#fastfood.append('Food Court')
#fastfood.append('Food Stands')
#fastfood.append('Food Trucks')
#fastfood.append('Gastropubs')
#fastfood.append('Sandwiches')
#fastfood.append('Tacos')
#fastfood.append('Ice Cream & Frozen Yogurt')


# In[459]:


amer_states=[]
amer_states.append('AZ')
amer_states.append('IL')
amer_states.append('NC')
amer_states.append('NV')
amer_states.append('NY')
amer_states.append('OH')
amer_states.append('PA')
amer_states.append('SC')
amer_states.append('VT')
amer_states.append('WI')


# In[460]:


ind=[]
for i in range(0,len(categories)):
    if categories[i] and set(categories[i]) & set(fastfood):
        ind.append(i)
        
stateind=[]
for i in range(0,len(yelpstates)):
    if yelpstates[i] in amer_states:
        stateind.append(i)

use_inds=list(set(stateind) & set(ind))


# In[461]:


bus_data_use=[]
for i in use_inds:
    bus_data_use.append(data[i])
bus_dict=dict()
for i in bus_data_use:
    bus_dict[i['business_id']]=i
    bus_id_list=[]
for i in bus_data_use: 
    bus_id_list.append(i['business_id'])


# In[38]:


print bus_data_use[0].keys()


# In[39]:


rev_counts=dict()
for i in bus_data_use:
    rev_counts[i['business_id']]=0
    


# In[462]:


print len(bus_data_use)


# In[ ]:



with open('yelp_academic_dataset_review.json') as f:
    for line in f: 
        rev=json.loads(line)
        if rev['business_id'] in bus_id_list:
            rev_counts[rev['business_id']]+=1
        rev=[]


# In[ ]:


data=[]
with open('yelp_academic_dataset_review.json') as f:
    for line in f: 
        rev=json.loads(line)
        if rev['business_id'] in bus_id_list: 
            data.append(rev)
        rev=[]


# In[40]:


import dill
#dill.dump(rev_counts, open('rest_reviews_counts.dill', 'w'))

data=dill.load(open('FF_reviews_onlyff.dill', 'r'))
rev_counts=dill.load(open('rest_reviews_counts.dill', 'r'))


# In[41]:


rev_dict=dict()
rev_dict['rating']=[]
#rev_dict['user_ids']=[]
#rev_dict['busid_rev']=[]
rev_dict['zipcode_rev']=[]
for i in data: 
    rev_dict['rating'].append(i['stars'])
#    rev_dict['zipcode_rev'].append(bus_dict[i['business_id']]['postal_code'])
#    user_ids.append(i['user_id'])
#    rev_dict['busid_rev'].append(i['business_id'])
    try:
        rev_dict['zipcode_rev'].append(int(bus_dict[i['business_id']]['postal_code'].strip()))
    except:
        rev_dict['zipcode_rev'].append(0)
    
#unique_UID=set(user_ids)


# In[451]:


print len(data)


# In[133]:


from textblob import TextBlob
sent_dict=dict()
for i in set(rev_dict['zipcode_rev']):
    sent_dict[i]=[]
    
for i in data: 
    try:
        sent_dict[int(bus_dict[i['business_id']]['postal_code'].strip())].append(TextBlob(i['text']).sentiment.polarity)
    except:
        pass

    


# In[ ]:





# In[ ]:





# In[ ]:





# In[42]:


rev_df=pd.DataFrame(rev_dict)


# In[43]:


print rev_df.head()


# In[44]:


get_ipython().magic(u'sql DROP TABLE IF EXISTS rev_df')
get_ipython().magic(u'sql PERSIST rev_df')


# In[45]:


get_ipython().run_cell_magic(u'sql', u'', u'select * from rev_df limit 10')


# In[46]:


get_ipython().run_cell_magic(u'sql', u'', u'select rating,zip,County,PCT_DIABETES_ADULTS10 from rev_df\njoin (SELECT zip,State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM \nzip_df JOIN (SELECT State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM health_data JOIN food_fda ON (FIPS=FIPS1))\nON (county2=County and state2=State)) on (zipcode_rev=zip) limit 10')


# In[47]:


A = get_ipython().magic(u'sql select rating,zip,County,PCT_DIABETES_ADULTS10 from rev_df join (SELECT zip,State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM zip_df JOIN (SELECT State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM health_data JOIN food_fda ON (FIPS=FIPS1)) ON (county2=County and state2=State)) on (zipcode_rev=zip) ')


# In[135]:


get_ipython().run_cell_magic(u'sql', u'', u'select zipcode_rev,count(*) as FFcount,AVG(rating) from rev_df group by zipcode_rev limit 5')


# In[49]:


print len(rev_counts.keys())


# In[50]:


revcount_dict=dict()
revcount_dict['zips']=[]
revcount_dict['counts']=[]
for i in rev_counts.keys():
    revcount_dict['zips'].append(bus_dict[i]['postal_code'])
    revcount_dict['counts'].append(rev_counts[i])
    


# In[51]:


revcount_df=pd.DataFrame(revcount_dict)


# In[52]:


get_ipython().magic(u'sql DROP TABLE IF EXISTS revcount_df')
get_ipython().magic(u'sql PERSIST revcount_df')


# In[134]:


get_ipython().run_cell_magic(u'sql', u'', u'select sum(counts),zips from revcount_df group by zips limit 5')


# In[70]:


get_ipython().run_cell_magic(u'sql', u'', u"select zips,(FFcount*1.0)/restcounts, avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join\n(select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev)\non(zips=zipcode_rev) limit 5")


# In[71]:


B = get_ipython().magic(u"sql select zips,(FFcount*1.0)/restcounts, avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join (select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev) on(zips=zipcode_rev) ")


# In[ ]:





# In[72]:


ff_ratio=[]
rating_av=[]
for i in B: 
    ff_ratio.append(i[1])
    rating_av.append(i[2])
    


# In[73]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# In[80]:


clf = Ridge(alpha=1)
data_train, data_test, perc_train, perc_test=train_test_split(np.array(ff_ratio).reshape(-1,1),rating_av,test_size=0.7)
clf.fit(data_train,perc_train)
pred=clf.predict(data_test)
    
print r2_score(perc_test,pred)


# In[ ]:





# In[139]:


for_df=dict()
for_df['zipcode']=[]
for_df['av_sent']=[]
for i in sent_dict.keys():
    if len(sent_dict[i])>0:
        for_df['zipcode'].append(i)
        for_df['av_sent'].append(sum(sent_dict[i])/len(sent_dict[i]))
sent_df=pd.DataFrame(for_df)    


# In[140]:


get_ipython().magic(u'sql DROP TABLE IF EXISTS sent_df')
get_ipython().magic(u'sql PERSIST sent_df')


# In[142]:


get_ipython().run_cell_magic(u'sql', u'', u'select * from sent_df limit 5')


# In[145]:


get_ipython().run_cell_magic(u'sql', u'', u"select zips,av_sent,FF_ratio,avrat from sent_df join\n(select zips,(FFcount*1.0)/restcounts as 'ff_ratio', avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join\n(select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev)\non(zips=zipcode_rev)) on (zipcode=zips) limit 5")


# In[146]:


C = get_ipython().magic(u"sql select zips,av_sent,FF_ratio,avrat from sent_df join (select zips,(FFcount*1.0)/restcounts as 'ff_ratio', avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join (select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev) on(zips=zipcode_rev)) on (zipcode=zips)")


# In[289]:


print len(C)


# In[153]:


yelpstats=dict()
yelpstats['av. sentiment']=[]
yelpstats['FF/rest. ratio']=[]
yelpstats['av. rating']=[]
for i in C:
    yelpstats['av. sentiment'].append(i[1])
    yelpstats['FF/rest. ratio'].append(i[2])
    yelpstats['av. rating'].append(i[3])
    


# In[154]:


yelpstats_df=pd.DataFrame(yelpstats)


# In[260]:


from pandas.plotting import scatter_matrix
scatter_matrix(yelpstats_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
#plt.show()
plt.savefig('fig5.png')


# In[ ]:





# In[ ]:





# In[82]:


revhealth=dict()
revhealth['rating']=[]
revhealth['diab_perc']=[]
for i in A: 
    revhealth['rating'].append(i[0])
    revhealth['diab_perc'].append(i[3])
    


# In[83]:


rat_enc=pd.get_dummies(revhealth['rating'])
print rat_enc.shape
print len(revhealth['rating'])


# In[84]:


ones=np.ones((len(revhealth['rating'])))
#VH=np.column_stack((VH,H))
#rat_enc.as_matrix()
predictor=np.column_stack((np.array(ones),revhealth['rating']))

glm_model=sm.GLM(revhealth['diab_perc'],predictor)
glm_results=glm_model.fit()
print glm_results.params,glm_results.pvalues


# In[85]:


print np.array(list(revhealth['diab_perc']))


# In[86]:


glm_results.summary()


# In[87]:


k=dict()
k[1]=[]
k[2]=[]
k[3]=[]
k[4]=[]
k[5]=[]

for n in range(1,6):
    for i in range(0,len(revhealth['rating'])):
        if revhealth['rating'][i] == n:
            k[n].append(float(revhealth['diab_perc'][i]))
            
            
        


# In[88]:


print k.keys()


# In[254]:


import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = ('1', '2', '3', '4', '5')
y_pos = np.arange(len(objects))

means=[]
for i in k.keys():
    means.append(sum(k[i])/len(k[i]))
se=[]
for i in k.keys():
    se.append(np.std(np.array(k[i]))/(len(k[i])**0.5))

plt.bar(y_pos, means, align='center', alpha=0.5)
plt.errorbar(y_pos, means, yerr=se, fmt='.')
plt.xticks(y_pos, objects)
plt.ylim([8.35,8.47])
plt.xlabel('Rating 1-5')
plt.ylabel('% Diabetic')
plt.title('Rating vs. Diabetes Rate')
#plt.show()
plt.savefig('fig3.png')


# In[231]:


revhealth_df=pd.DataFrame(revhealth)


# In[257]:


ax = sns.regplot(x="rating", y="diab_perc", data=revhealth_df, x_estimator=np.mean, logx=True)
plt.savefig('fig4.png')
#plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


yelpstats['av. sentiment']
yelpstats['FF/rest. ratio']
yelpstats['av. rating']


# In[158]:


from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor


# In[163]:


clf = Ridge(alpha=1)
data_train, data_test, perc_train, perc_test=train_test_split(np.array(yelpstats['av. sentiment']).reshape(-1,1),np.array(yelpstats['av. rating']).reshape(-1,1),test_size=0.7)
clf.fit(data_train,perc_train)
pred=clf.predict(data_test)
    
print r2_score(perc_test,pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[261]:


get_ipython().run_cell_magic(u'sql', u'', u'select rating,zip,County,PCT_DIABETES_ADULTS10 from rev_df\njoin (SELECT zip,State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM \nzip_df JOIN (SELECT State,County,PCT_DIABETES_ADULTS10,FFRPTH12 FROM health_data JOIN food_fda ON (FIPS=FIPS1))\nON (county2=County and state2=State)) on (zipcode_rev=zip) limit 5')


# In[262]:


import csv


# In[263]:


#http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0111119
#Article Source: Local Population Characteristics and Hemoglobin A1c Testing Rates among Diabetic Medicare Beneficiaries 
#Yasaitis LC, Bubolz T, Skinner JS, Chandra A (2014) Local Population Characteristics and Hemoglobin A1c Testing Rates among Diabetic Medicare Beneficiaries. PLOS ONE 9(10): e111119. https://doi.org/10.1371/journal.pone.0111119

medi_data=[]
with open('diabetes_by_zc.txt') as f: 
    reader = csv.reader(f, delimiter=',')
    for line in reader: 
        medi_data.append(line)
        


# In[277]:


print medi_data[0]


# In[313]:


medi_dict=dict()
medi_dict['medi_zip']=[]
medi_dict['bene_rat']=[]
medi_dict['poverty']=[]
medi_dict['ed']=[]
for i in medi_data:
    if i[0]=='zipcode':
        continue
    if i[6] and i[5]:
        if len(i[0].strip())==4:
            medi_dict['medi_zip'].append('0'+i[0].strip())
        else:
            medi_dict['medi_zip'].append(i[0].strip())
            
        medi_dict['poverty'].append(float(i[2]))
        medi_dict['ed'].append(float(i[1]))
        medi_dict['bene_rat'].append(float(i[6])/float(i[5]))
    


# In[314]:


medi_df=pd.DataFrame(medi_dict)


# In[315]:


get_ipython().magic(u'sql DROP TABLE IF EXISTS medi_df')
get_ipython().magic(u'sql PERSIST medi_df')


# In[316]:


get_ipython().run_cell_magic(u'sql', u'', u'select * from medi_df limit 5')


# In[450]:


get_ipython().run_cell_magic(u'sql', u'', u"select zips,av_sent as 'avg_sentiment',avrat as 'avg_rating',FF_ratio from sent_df join\n(select zips,(FFcount*1.0)/restcounts as 'ff_ratio', avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join\n(select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev)\non(zips=zipcode_rev)) on (zipcode=zips) limit 15")


# In[447]:


get_ipython().run_cell_magic(u'sql', u'', u"select * from medi_df join \n\n(select zips,av_sent,FF_ratio,avrat from sent_df join\n(select zips,(FFcount*1.0)/restcounts as 'ff_ratio', avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join\n(select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev)\non(zips=zipcode_rev)) on (zipcode=zips)) on (medi_zip=zips)\n\nlimit 15")


# In[319]:


D = get_ipython().magic(u"sql select * from medi_df join (select zips,av_sent,FF_ratio,avrat from sent_df join (select zips,(FFcount*1.0)/restcounts as 'ff_ratio', avrat from (select sum(counts) as 'restcounts',zips from revcount_df group by zips) join (select zipcode_rev,count(*) as 'FFcount',AVG(rating) as 'avrat' from rev_df group by zipcode_rev) on(zips=zipcode_rev)) on (zipcode=zips)) on (medi_zip=zips)")


# In[320]:


print len(D)


# In[321]:


ym_dict=dict()
ym_dict['bene_ratio']=[]
ym_dict['edu']=[]
ym_dict['pov']=[]
ym_dict['senti']=[]
ym_dict['FF_ratio']=[]
ym_dict['av_rating']=[]

for i in D: 
    ym_dict['bene_ratio'].append(i[1])
    ym_dict['edu'].append(i[2])
    ym_dict['pov'].append(i[4])
    ym_dict['senti'].append(i[6])
    ym_dict['FF_ratio'].append(i[7])
    ym_dict['av_rating'].append(i[8])


# In[322]:


VH=np.column_stack((np.array(ym_dict['av_rating']),np.array(ym_dict['edu']),np.array(ym_dict['pov']),np.array(ym_dict['senti']),np.array(ym_dict['FF_ratio'])))


# In[413]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor


# In[463]:


sc = StandardScaler()
clf = Ridge(alpha=10)
#Ridge(alpha=10)
#RandomForestRegressor(min_samples_split=100,n_estimators=150)
#Ridge(alpha=10)
data_train, data_test, perc_train, perc_test=train_test_split(sc.fit_transform(VH),ym_dict['bene_ratio'],test_size=0.5)
clf.fit(data_train,perc_train)
pred=clf.predict(data_test)
    
print r2_score(perc_test,pred)
print(cross_val_score(clf, VH, ym_dict['bene_ratio'],cv=5,scoring='r2')) 


# In[467]:


plt.plot(pred,perc_test,'.')
plt.xlabel('Prediction')
plt.ylabel('Actual')
#plt.show()
plt.savefig('fig6.png')


# In[392]:


from sklearn import base


# In[394]:


class ResidualEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self,lin_est,nlin_est):
        
        self.lin_est=lin_est
        self.nlin_est=nlin_est
        
    def fit(self, X, y):
        
        self.lin_fit=self.lin_est.fit(X,y)
        self.residuals=np.array(y)-self.lin_est.predict(X)
        self.nlin_fit=self.nlin_est.fit(X,self.residuals)
        
    def predict(self, X):
        return self.lin_est.predict(X)+self.nlin_est.predict(X)


# In[423]:


data_train, data_test, perc_train, perc_test=train_test_split(sc.fit_transform(VH),ym_dict['bene_ratio'],test_size=0.5)
regressor = RandomForestRegressor(min_samples_split=100,n_estimators=150)
dec_tree=DecisionTreeRegressor(max_depth=5)
clf = Ridge(alpha=10)
resest=ResidualEstimator(clf,regressor)
resest.fit(data_train,perc_train)
pred=resest.predict(data_test)
print r2_score(perc_test,pred)
print(cross_val_score(resest, VH, ym_dict['bene_ratio'],cv=5,scoring='r2')) 


# In[422]:


plt.plot(pred,perc_test,'.')
plt.show()


# In[424]:


ym_df=pd.DataFrame(ym_dict)
scatter_matrix(ym_df, alpha=0.2, figsize=(6, 6), diagonal='kde')
plt.show()


# In[ ]:





# In[444]:


#import statsmodels.api as sm
predictor=sm.add_constant(VH)

print predictor.shape,np.transpose(np.array(ym_dict['bene_ratio'])).shape 
glm_model=sm.GLM(np.array(ym_dict['bene_ratio']),predictor)
glm_results=glm_model.fit()
print glm_results.params,glm_results.pvalues


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




