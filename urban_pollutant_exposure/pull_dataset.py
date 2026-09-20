import requests
import pandas as pd

CENSUS_API_KEY = "7325ebcab86ef8d18acce7f121da552d5c36e96e"

url = "https://api.census.gov/data/2022/acs/acs5"
params = {
    "get": "NAME,B19013_001E",
    "for": "zip code tabulation area:*",
    "key": CENSUS_API_KEY
}
resp = requests.get(url, params=params)
print(resp.status_code)

data = resp.json()
census_df = pd.DataFrame(data[1:], columns=data[0])
census_df.rename(columns={
    "zip code tabulation area": "zip_code",
    "B19013_001E": "median_household_income"
}, inplace=True)
census_df["median_household_income"] = pd.to_numeric(census_df["median_household_income"], errors="coerce")
census_df = census_df[census_df["median_household_income"] > 0]  # drops -666666666 placeholder nulls
census_df.to_csv("acs_median_income_by_zip.csv", index=False)