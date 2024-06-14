#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install dash plotly
server = app.server

# In[2]:


import os
os.getcwd()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('Alpha')


# In[6]:


# Removing outliers of salary


# In[8]:


df= df[df['medium_salary'] <= 83000]


# In[14]:


# saving the dataframe
df.to_csv('my_data.csv')


# In[16]:


df.drop(columns=['Unnamed: 0'], inplace=True)


# In[18]:


df.info()


# In[20]:


df.nunique()


# In[22]:


# creating dataframe for decision tree regression


# In[24]:


cols_to_drop = ['job_id', 'company_id', 'maximum_salary', 'minimum_salary',
                'state', 'city', 'industry_id', 'job_title', 'description',
                'location', 'employee_count',]


# In[26]:


df = df.drop(columns=cols_to_drop)


# In[28]:


df.info()


# In[30]:


# Ranking the experience level so it makes it easier to for the regression to work instead or random assignment


# In[32]:


# Define the mapping
experience_level_mapping = {
    'Internship': 1,
    'Entry level': 2,
    'Associate': 3,
    'Mid-Senior level': 4,
    'Director': 5,
    'Executive': 6
}

# Apply the mapping
df['formatted_experience_level'] = df['formatted_experience_level'].map(experience_level_mapping)

# Verify the changes
print(df['formatted_experience_level'].head())


# In[34]:


#Changing them into dummy variables
df_dummies = pd.get_dummies(df, drop_first=True)


# In[36]:


df_dummies.info()


# In[38]:


X = df_dummies.drop(columns='medium_salary')
y = df_dummies['medium_salary']


# In[40]:


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Create a Decision Tree Regressor object
regressor = DecisionTreeRegressor(random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Save the fitted model
joblib.dump(regressor, 'decision_tree_regressor.pkl')

# Predict on the test data
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('decision_tree_regressor.pkl')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Salary Prediction Dashboard"),
    html.Div([
        html.Label("Experience Level"),
        dcc.Dropdown(
            id='formatted_experience_level_dropdown',
            options=[{'label': level, 'value': level} for level in df['formatted_experience_level'].unique()],
            placeholder='Select Experience Level'
        ),
        html.Label("Industry Group"),
        dcc.Dropdown(
            id='group_industry_dropdown',
            options=[{'label': industry, 'value': industry} for industry in df['group_industry'].unique()],
            placeholder='Select Industry',
        ),
        html.Label("Category"),
        dcc.Dropdown(
            id='category_dropdown',
            options=[{'label': category, 'value': category} for category in df['category'].unique()],
            placeholder='Select Category',
        ),
        html.Button('Predict Salary', id='predict_button', n_clicks=0),
        html.Div(id='output_salary')
    ], style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='salary_graph')
])

# Define the callback for prediction
@app.callback(
    Output('output_salary', 'children'),
    [Input('predict_button', 'n_clicks')],
    [dash.dependencies.State('formatted_experience_level_dropdown', 'value'),
     dash.dependencies.State('group_industry_dropdown', 'value'),
     dash.dependencies.State('category_dropdown', 'value')]
)
def update_output(n_clicks, experience_level, industry, category):
    if n_clicks > 0 and experience_level and industry and category:
        # Create a new DataFrame with the selected values
        input_data = pd.DataFrame([[experience_level]], columns=['formatted_experience_level'])
        input_data = pd.concat([input_data, pd.get_dummies(pd.DataFrame([[industry]], columns=['group_industry']), drop_first=True)], axis=1)
        input_data = pd.concat([input_data, pd.get_dummies(pd.DataFrame([[category]], columns=['category']), drop_first=True)], axis=1)

        # Align the columns of the input data with the training data
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Predict the salary using the loaded model
        predicted_salary = model.predict(input_data)[0]

        return f"Predicted Medium Salary: ${predicted_salary:.2f}"
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)

# In[42]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('decision_tree_regressor.pkl')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("Salary Prediction Dashboard"),
    html.Div([
        html.Label("Experience Level"),
        dcc.Dropdown(
            id='formatted_experience_level_dropdown',
            options=[{'label': level, 'value': level} for level in df['formatted_experience_level'].unique()],
            placeholder='Select Experience Level'
        ),
        html.Label("Industry Group"),
        dcc.Dropdown(
            id='group_industry_dropdown',
            placeholder='Select Industry',
        ),
        html.Label("Category"),
        dcc.Dropdown(
            id='category_dropdown',
            placeholder='Select Category',
        ),
        html.Button('Predict Salary', id='predict_button', n_clicks=0),
        html.Div(id='output_salary')
    ], style={'width': '50%', 'display': 'inline-block'}),
    dcc.Graph(id='salary_graph')
])


# In[44]:


@app.callback(
    Output('group_industry_dropdown', 'options'),
    [Input('formatted_experience_level_dropdown', 'value')]
)
def set_industry_options(selected_experience_level):
    if not selected_experience_level:
        return []
    filtered_df = df[df['formatted_experience_level'] == selected_experience_level]
    industries = filtered_df['group_industry'].unique()
    return [{'label': industry, 'value': industry} for industry in industries]

@app.callback(
    Output('category_dropdown', 'options'),
    [Input('group_industry_dropdown', 'value'),
     Input('formatted_experience_level_dropdown', 'value')]
)
def set_category_options(selected_industry, selected_experience_level):
    if not selected_experience_level or not selected_industry:
        return []
    filtered_df = df[(df['formatted_experience_level'] == selected_experience_level) & 
                     (df['group_industry'] == selected_industry)]
    categories = filtered_df['category'].unique()
    return [{'label': category, 'value': category} for category in categories]

@app.callback(
    Output('output_salary', 'children'),
    [Input('predict_button', 'n_clicks')],
    [dash.dependencies.State('formatted_experience_level_dropdown', 'value'),
     dash.dependencies.State('group_industry_dropdown', 'value'),
     dash.dependencies.State('category_dropdown', 'value')]
)
def update_output(n_clicks, experience_level, industry, category):
    if n_clicks > 0 and experience_level and industry and category:
        # Create a new DataFrame with the selected values
        input_data = pd.DataFrame([[experience_level]], columns=['formatted_experience_level'])
        input_data = pd.concat([input_data, pd.get_dummies(pd.DataFrame([[industry]], columns=['group_industry']), drop_first=True)], axis=1)
        input_data = pd.concat([input_data, pd.get_dummies(pd.DataFrame([[category]], columns=['category']), drop_first=True)], axis=1)

        # Align the columns of the input data with the training data
        input_data = input_data.reindex(columns=X.columns, fill_value=0)

        # Predict the salary using the loaded model
        predicted_salary = model.predict(input_data)[0]

        return f"Predicted Monthly Salary: ${predicted_salary:.2f}"
    return ""

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8060)


# In[ ]:




