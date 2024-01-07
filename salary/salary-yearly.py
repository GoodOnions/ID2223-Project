import pandas as pd
import pycountry_convert as pc
import requests
import datetime
import hopsworks

def fetch_data():
    year = datetime.date.today().year
    data = pd.read_csv("https://ai-jobs.net/salaries/download/salaries.csv")
    data = data[data["work_year"] == year]
    return data

def prepare_data(dataset: pd.DataFrame):
    ## We need to apply the same transformations used during the EDA

    quantile = dataset['salary_in_usd'].quantile(0.99)
    dataset = dataset[dataset['salary_in_usd'] < quantile]
    
    dataset['salary'] = dataset['salary_in_usd']
    dataset.drop(columns=['salary_in_usd','employee_residence', 'remote_ratio'], inplace=True)

    salary_currency_list = [
        'USD',
        'EUR',
        'GBP'
    ]
    dataset['salary_currency'] = dataset['salary_currency'].map(lambda x: x if x in salary_currency_list else 'other_currency')
    dataset = pd.get_dummies(dataset, columns=['salary_currency'],prefix='', prefix_sep='', sparse=False).drop(columns=['other_currency'])

    experience_level_map = {
        'EN': 0,
        'MI': 1,
        'SE': 2,
        'EX': 3
    }
    dataset['experience_level'] = dataset['experience_level'].map(experience_level_map)

    jobs = dataset['job_title'].value_counts()
    jobs = jobs[jobs.cumsum() < len(dataset) * 0.9]
    jobs = jobs.index.tolist()
    dataset = dataset[dataset['job_title'].isin(jobs)]

    dataset = dataset[dataset['employment_type'] == 'FT'].drop(columns=['employment_type'])
    dataset['Engineer'] = dataset['job_title'].apply(lambda x: 1 if 'engineer' in x.lower() else 0)
    dataset['Scientist'] = dataset['job_title'].apply(lambda x: 1 if 'data scientist' in x.lower() else 0)
    dataset['Research'] = dataset['job_title'].apply(lambda x: 1 if 'research' in x.lower() else 0)
    dataset['Analyst'] = dataset['job_title'].apply(lambda x: 1 if 'analyst' in x.lower() else 0)
    dataset = pd.get_dummies(dataset, columns=['job_title'],prefix='', prefix_sep='', sparse=False)

    company_size_map = {
        'S': 0,
        'M': 1,
        'L': 2,
    }
    dataset['company_size'] = dataset['company_size'].map(company_size_map)

    dataset['company_continent'] = dataset['company_location'].apply(lambda x: pc.country_alpha2_to_continent_code(x))
    dataset = dataset[dataset['company_continent'].isin(['EU', 'NA'])]

    country_CPI = dataset[['work_year','company_location']].drop_duplicates()
    country_CPI['CPI'] = country_CPI.apply(lambda x: get_gdp_by_country_code(x['company_location'], x['work_year']), axis=1)
    country_GDP = dataset[['work_year','company_location']].drop_duplicates()
    country_GDP['GDP'] = country_CPI.apply(lambda x: get_gdp_by_country_code(x['company_location'], x['work_year'], index='NY.GDP.MKTP.CD'), axis=1)
    dataset_final = dataset.merge(country_GDP, on=['company_location','work_year'], how='left')
    dataset_final = dataset_final.merge(country_CPI, on=['company_location','work_year'], how='left')
    dataset_final = dataset_final.drop(columns=['company_location', 'company_continent'])

    salary_bins = pd.qcut(dataset_final['salary'], q=3, labels=False)
    dataset_final['salary_bins'] = salary_bins

    return dataset_final

def upload_data(df: pd.DataFrame):
    project = hopsworks.login()
    fs = project.get_feature_store()

    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(' ', '_')
    primary_key = df.columns[df.columns != 'salary_bins'].values

    salary_fg = fs.get_or_create_feature_group(
        name="salary",
        version=1,
        primary_key=primary_key,
        description="salary dataset")
    
    salary_fg.insert(df)
    return

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

def main():
    print("Fetching data...")
    data = fetch_data()
    print(f"Done. (Fetched {len(data)} rows)")

    if len(data) <= 0:
        print("No new data fetched.")
        return
    
    print("Preparing data...")
    data = prepare_data(data)
    print("Done.")
    print("Uploading data...")
    upload_data(data)
    print("Done.")

if __name__ == "__main__":
    main()