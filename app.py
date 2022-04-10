#!/usr/bin/env python
# coding: utf-8

# In[61]:


from flask import Flask


# In[62]:


app = Flask(__name__)


# In[63]:


import joblib


# In[64]:


model_cart = joblib.load('group4_prj_cc_fraud_model_cart')


# In[65]:


model_rf = joblib.load('group4_prj_cc_fraud_model_rf')


# In[66]:


model_boost = joblib.load('group4_prj_cc_fraud_model_gb')


# In[67]:


model_logreg = joblib.load('group4_prj_cc_fraud_model_lr')


# In[68]:


from flask import request, render_template


# In[69]:


@app.route("/", methods=["GET","POST"])

def index():
    if request.method == "POST":
        cc_num = float(request.form.get("cc_num"))
        amount = float(request.form.get("amount"))
        gender = float(request.form.get("gender"))        
        zip = float(request.form.get("zip"))
        city_pop = float(request.form.get("city_pop"))
        unixtime = float(request.form.get("unixtime"))
        pur_lat = float(request.form.get("pur_lat"))
        pur_long = float(request.form.get("pur_long"))
        merc_lat = float(request.form.get("merc_lat"))
        merc_long = float(request.form.get("merc_long"))
        lat_diff = abs(pur_lat - merc_lat)
        long_diff = abs(pur_long - merc_long)
        category_code = float(request.form.get("category_code"))
         
        print(cc_num)
        print(amount)
        print(gender)
        print(zip)
        print(city_pop)
        print(unixtime)
        print(lat_diff)
        print(long_diff)
        print(category_code)
        
        res_cart = model_cart.predict([[cc_num, amount, gender, zip, city_pop, unixtime, lat_diff, long_diff, category_code]])
        res_rf = model_rf.predict([[cc_num, amount, gender, zip, city_pop, unixtime, lat_diff, long_diff, category_code]])
        res_gb = model_boost.predict([[cc_num, amount, gender, zip, city_pop, unixtime, lat_diff, long_diff, category_code]])
        res_logreg = model_logreg.predict([[cc_num, amount, gender, zip, city_pop, unixtime, lat_diff, long_diff, category_code]])
        
        return(render_template("index.html", result_cart=res_cart, result_rf=res_rf, result_gb=res_gb, result_logreg=res_logreg))
    else:
        return(render_template("index.html", result_cart="Loaded", result_rf="Loaded", result_gb="Loaded", result_logreg="Loaded"))
    


# In[70]:


if __name__ == "__main__":
    app.run()


# In[ ]:




