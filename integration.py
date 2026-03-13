from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time

geolocator = Nominatim(user_agent='urban_pollution_study_v1')
geocode    = RateLimiter(geolocator.reverse, min_delay_seconds=1)

def get_zip(lat, lon):
    try:
        location = geocode(f'{lat}, {lon}', exactly_one=True,
                           language='en')
        if location and 'postcode' in location.raw.get('address', {}):
            return str(location.raw['address']['postcode']).zfill(5)[:5]
    except Exception:
        pass
    return None

unique_sensors = aq_df[['locationId', 'latitude', 'longitude']].drop_duplicates()

unique_sensors['zip_code'] = unique_sensors.apply(
    lambda row: get_zip(row['latitude'], row['longitude']), axis=1
)

aq_df = aq_df.merge(unique_sensors[['locationId', 'zip_code']],
                    on='locationId', how='left')

zip_agg = (
    aq_df.groupby('zip_code')['pm25_ug_m3']
    .agg(mean_pm25='mean', sensor_count='count', std_pm25='std')
    .reset_index()
)

unified_df = zip_agg.merge(census_df[['zip_code', 'median_income_usd', 'state']],
                            on='zip_code', how='inner')

unified_df['income_quintile'] = pd.qcut(
    unified_df['median_income_usd'],
    q=5,
    labels=[1, 2, 3, 4, 5]
).astype(int)

unified_df.dropna(subset=['mean_pm25', 'median_income_usd'], inplace=True)
unified_df.reset_index(drop=True, inplace=True)

print(unified_df[['zip_code', 'mean_pm25', 'median_income_usd',
                   'income_quintile', 'state']].head(10).to_string())
print(f'\nFinal unified dataset shape: {unified_df.shape}')

unified_df.to_csv('unified_pollution_income.csv', index=False)
print('Dataset saved to unified_pollution_income.csv')