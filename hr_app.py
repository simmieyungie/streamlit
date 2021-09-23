"""
Created on Mon Sep 20 00:11:37 2021
@author: SIMIYOUNG
"""

import streamlit as st
import pandas as pd

#encoder
#encodder
def encoder(x):
    if x == True:
        return 1
    else:
        return 0

#header and title
st.title("HR Predictive Model")

#side bar form
with st.form(key = "features"):
    with st.sidebar:
        #read in dataset
        df = pd.read_csv("https://raw.githubusercontent.com/simmieyungie/HR-Analytics/master/attrition.csv")
        
        st.sidebar.header("Select input features for sidebar")
        #side bar title
        
        #enter satisfaction level score
        st.sidebar.subheader("Satistfaction Level")
        satisfaction = st.sidebar.number_input("Enter satisfaction score")
        st.sidebar.write("The current satisfaction is", round(satisfaction, 2))
        
        #enter satisfaction level score
        st.sidebar.subheader("Last Evaluation")
        evaluation = st.sidebar.number_input("Enter Evaluation score")
        
        #number of projects taken
        st.sidebar.subheader("Number of Projects")
        projects = st.sidebar.number_input("Projects undertaken", min_value = 0, max_value = 10, step = 1)
        
        
        #average monthly hours
        st.sidebar.subheader("Monthly Hours")
        hours = st.sidebar.number_input("Average Monthlyh Hours", 
                                        min_value = df.average_montly_hours.min(), max_value = df.average_montly_hours.max(), step = 1)
        
        #time spent in company
        st.sidebar.subheader("Time spent in company")
        time_spent = st.sidebar.number_input("Time in company", min_value = 0, max_value = df.time_spend_company.max(), step = 1)
        
        #Work accident
        st.sidebar.subheader("Work Accident")
        accident = st.sidebar.checkbox("Work Accident?")
        
        #little message to affirm work accident or not
        if accident == True:
            st.sidebar.write("Yes, workaccident")
        else:
            st.sidebar.write("No workaccident")
        
        accident = encoder(accident)
        
        
        #promotion
        st.sidebar.subheader("Promotion in five years")
        promotion = st.sidebar.checkbox("Promotion?")
        
        #encode promotion
        promotion = encoder(promotion)
        
        #department
        st.sidebar.subheader("Department")
        
        #get departments
        dept_list =  list(df.dept.unique())#.sort(reverse=True)
        dept_list.sort()
        dept = st.sidebar.selectbox("Select department", options = dept_list) #come back to sort it
        
        #get salaryt
        st.sidebar.subheader("Salary")
        #get salary
        sal_list = list(df.salary.unique())#.sort(reverse=True)
        
        salary = st.sidebar.selectbox("Select Salary Range", options = sal_list) #come back to sort it
        
        #predict button
        feature_run = st.form_submit_button("Run")
        
        #create dataframe of objects
data = st.container()


with data:
    if feature_run:
        input_features = {'satisfaction_level' : satisfaction, 'last_evaluation' : evaluation, 
                  'number_project' : projects, 'average_montly_hours' : 
                  hours, 'time_spend_company' : time_spent, 'Work_accident' : accident, 
                  'promotion_last_5years' : promotion, 'dept' : dept, 'salary' : salary}
        
        inputs = pd.DataFrame(input_features, index=[0])
        st.write(inputs)
