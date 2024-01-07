import gradio as gr
import hopsworks
import joblib
import pandas as pd

features =  ['work_year',
             'experience_level',
             'company_size',
             'eur',
             'gbp',
             'usd',
             'engineer',
             'scientist',
             'research',
             'analyst',
             'analytics_engineer',
             'applied_scientist',
             'bi_developer',
             'business_intelligence_analyst',
             'business_intelligence_engineer',
             'data_analyst',
             'data_architect',
             'data_engineer',
             'data_manager',
             'data_science_consultant',
             'data_science_manager',
             'data_scientist',
             'ml_engineer',
             'machine_learning_engineer',
             'machine_learning_scientist',
             'research_analyst',
             'research_engineer',
             'research_scientist',
             'gdp',
             'cpi']


labels = ['(16454.999, 122000.0]', '(122000.0, 170000.0]', '(170000.0, 329700.0]']

project = hopsworks.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("salary_model", version=5)
model_dir = model.download()
model = joblib.load(model_dir + "/model.pkl")
print("Model downloaded")

import requests

def get_gdp_by_country_code(country_code, year=2023, index='FP.CPI.TOTL'):
    # World Bank API endpoint for GDP data
    api_url = f'http://api.worldbank.org/v2/country/{country_code}/indicator/{index}?data={year}&format=json'


    # Make a GET request to the API
    response = requests.get(api_url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Extract the GDP value from the response
        gdp_value = data[1][0]['value'] if data[1] else None

        return gdp_value
    else:
        # If the request was not successful, print an error message
        print(f"Error: Unable to fetch data. Status code: {response.status_code}")
        return None

def salary(work_year,
             experience_level,
             company_size,
             currency,
             job_title,
             country)-> str:


    jobs =   ['analytics_engineer',
             'applied_scientist',
             'bi_developer',
             'business_intelligence_analyst',
             'business_intelligence_engineer',
             'data_analyst',
             'data_architect',
             'data_engineer',
             'data_manager',
             'data_science_consultant',
             'data_science_manager',
             'data_scientist',
             'ml_engineer',
             'machine_learning_engineer',
             'machine_learning_scientist',
             'research_analyst',
             'research_engineer',
             'research_scientist']

    jobs_flag ={}

    for name in jobs:
        if name == job_title.lower().replace(' ', '_'):
            jobs_flag[name] = True
        else:
            jobs_flag[name] = False

    role = [
        'engineer',
        'scientist',
        'research',
        'analyst'
    ]

    role_flag = {}

    for name in role:
        if name in job_title.lower():
            role_flag[name]= True
        else:
            role_flag[name] = False

    currency_flag = {
        'eur': False,
        'gbp': False,
        'usd': False
    }

    currency_flag[currency.lower()] = True

    company_size_dic = {
        'S': 0,
        'M': 1,
        'L': 2,
    }


    experience_level_map = {
        'EN': 0,
        'MI': 1,
        'SE': 2,
        'EX': 3
    }





    params = {}
    params['work_year'] = work_year
    params['experience_level'] = experience_level_map[experience_level]
    params['company_size'] = company_size_dic[company_size]
    params.update(currency_flag)
    params.update(role_flag)
    params.update(jobs_flag)
    params['gdp'] = get_gdp_by_country_code(country, work_year, 'NY.GDP.MKTP.CD')
    params['cpi'] = get_gdp_by_country_code(country, work_year, 'FP.CPI.TOTL')



    df = pd.DataFrame([params])
    print("Predicting")
    print(df)
    print(df.columns)

    res = model.predict(df)

    print(f"{labels[res[0]]} $")
    return f"{labels[res[0]]} $"

job_title_options = ['analytics_engineer',
             'applied_scientist',
             'bi_developer',
             'business_intelligence_analyst',
             'business_intelligence_engineer',
             'data_analyst',
             'data_architect',
             'data_engineer',
             'data_manager',
             'data_science_consultant',
             'data_science_manager',
             'data_scientist',
             'ml_engineer',
             'machine_learning_engineer',
             'machine_learning_scientist',
             'research_analyst',
             'research_engineer',
             'research_scientist']

demo = gr.Interface(
    fn=salary,
    title="Salary prediction",
    description="Prediction of the salary in USD",
    allow_flagging="never",
    inputs=[
        gr.components.Number(label='work_year'),
        gr.components.Radio(label='experience_level', choices=['EN', 'MI', 'SE', 'EX']),
        gr.components.Radio(label='company_size', choices=['S', 'M', 'L']),
        gr.components.Radio(label='currency', choices=['EUR', 'GBP', 'USD']),
        gr.components.Dropdown(label='job_title', choices=job_title_options),
        gr.components.Textbox(label='country', info='2 letter code', value='US')
    ],
    outputs=gr.Text())

demo.launch(debug=True, share=True)
