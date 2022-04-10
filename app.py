#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


import joblib


# In[4]:


model_cart = joblib.load('group4_prj_cc_fraud_model_cart')


# In[5]:


model_rf = joblib.load('group4_prj_cc_fraud_model_rf')


# In[6]:


model_boost = joblib.load('group4_prj_cc_fraud_model_gb')


# In[7]:


model_logreg = joblib.load('group4_prj_cc_fraud_model_lr')


# In[8]:


from flask import request, render_template


# In[9]:


def map_age_group(age):    
    if age <= 19:
        return 0 #teenage
    elif age > 19 and age <= 24:
        return 1 #young_adult
    elif age > 24 and age <= 39:
        return 2 #adult
    elif age> 39 and age <= 54:
        return 3 #middle_aged
    else:
        return 4 #elderly


# In[10]:


def map_price_group(amt):    
    if amt <= 30:
        return 0 #Cheap
    elif amt > 30 and amt <= 60:
        return 1 #Affordable
    elif amt > 60 and amt <= 499:
        return 2 #Average
    elif amt > 499 and amt <= 2999:
        return 3 #Expensive
    else:
        return 4 #Luxury


# In[11]:


import numpy as np


# In[12]:


@app.route("/", methods=["GET","POST"])

def index():
    if request.method == "POST":
        amount = map_price_group( float(request.form.get("amount")) )
        gender = float(request.form.get("gender"))        
        category_code = float(request.form.get("category_code"))

        age = map_age_group( float(request.form.get("age")) )
        
        pur_lat = float(request.form.get("pur_lat"))
        pur_long = float(request.form.get("pur_long"))
        merc_lat = float(request.form.get("merc_lat"))
        merc_long = float(request.form.get("merc_long"))
        lat_diff = abs(pur_lat - merc_lat)
        long_diff = abs(pur_long - merc_long)
        distance = np.linalg.norm([long_diff, lat_diff])
                                      
        print('amount=> ', amount, ' -> ', float(request.form.get("amount")))
        print('gender=>', gender)
        print('category_code=>', category_code)
        print('age=>', age)
        print('lat_diff=>', lat_diff)
        print('long_diff=>', long_diff)
        print('distance', distance)
        
        #Model Order: gender,age_group,price_range,distance,category_code
        
        res_cart = model_cart.predict([[gender, age, amount, distance, category_code]])
        res_cart = "Fraud" if res_cart==1 else "Not Fraud"
        res_rf = model_rf.predict([[gender, age, amount, distance, category_code]])
        res_rf = "Fraud" if res_rf==1 else "Not Fraud"
        res_gb = model_boost.predict([[gender, age, amount, distance, category_code]])
        res_gb = "Fraud" if res_gb==1 else "Not Fraud"
        res_logreg = model_logreg.predict([[gender, age, amount, distance, category_code]])
        res_logreg = "Fraud" if res_logreg==1 else "Not Fraud"
        
        return(render_template("index.html", result_cart=res_cart, result_rf=res_rf, result_gb=res_gb, result_logreg=res_logreg))
    else:
        return(render_template("index.html", result_cart="Loaded", result_rf="Loaded", result_gb="Loaded", result_logreg="Loaded"))
    


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




