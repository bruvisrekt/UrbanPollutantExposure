import requests
import pandas as pd
import numpy as np

OPENAQ_URL = 'https://api.openaq.org/v3/measurements'

params = {
    'parameter_id': 2,
    'country_id': 'US',
    'limit': 10000,
    'date_from': '2023-01-01',
    'date_to':   '2024-01-01',
}

headers = {'X-API-Key': 'YOUR_OPENAQ_API_KEY'}

response = requests.get(OPENAQ_URL, params=params, headers=headers)
response.raise_for_status()

raw_aq = pd.DataFrame(response.json()['results'])

raw_aq['latitude']  = raw_aq['coordinates'].apply(
    lambda x: x['latitude']  if isinstance(x, dict) else np.nan
)
raw_aq['longitude'] = raw_aq['coordinates'].apply(
    lambda x: x['longitude'] if isinstance(x, dict) else np.nan
)
raw_aq['timestamp'] = raw_aq['date'].apply(
    lambda x: x['utc']       if isinstance(x, dict) else np.nan
)

aq_df = raw_aq[['locationId', 'location', 'latitude',
                'longitude', 'timestamp', 'value']].copy()
aq_df.rename(columns={'value': 'pm25_ug_m3'}, inplace=True)

aq_df = aq_df[aq_df['pm25_ug_m3'] >= 0]
aq_df = aq_df[aq_df['pm25_ug_m3'] <= 500]

census_df = pd.read_csv('acs_median_income_by_zip.csv', dtype={'zip_code': str})

census_df.rename(columns={
    'zip_code':                'zip_code',
    'median_household_income': 'median_income_usd'
}, inplace=True)

census_df.dropna(subset=['median_income_usd'], inplace=True)
census_df['zip_code'] = census_df['zip_code'].str.zfill(5)
census_df = census_df[census_df['median_income_usd'] > 0]

print(f'AQ records after cleaning : {len(aq_df):,}')
print(f'ZIP codes in Census data  : {len(census_df):,}')