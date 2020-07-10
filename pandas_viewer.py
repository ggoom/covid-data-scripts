import pandas as pd
import numpy as np
import math

df = pd.read_csv("us-counties.csv", parse_dates=['date'])
df['fips'] = df.apply(lambda row: 'NYC' if math.isnan(row['fips']) and row['county'] == 'New York City' else row['fips'], axis=1)
df = df.set_index(['fips', 'date']).sort_index()


df = df[df['fips'] == 'NYC']
print(df)
