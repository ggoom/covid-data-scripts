import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np

# Load raw mask data & intervention data & county-level election data
df = pd.read_csv("masks/mask-prevalence.csv", index_col=0)
df = pd.melt(df, id_vars=['State'], value_vars=['2020-04-21', '2020-05-23'])
df = df.rename(columns={'variable': 'date', 'value': 'percent_wearing_masks'})
interventions = pd.read_csv("masks/interventions.csv").groupby('STATE').mean(numeric_only=True).round().fillna('737500')
counties = pd.read_csv('masks/2016_US_County_Level_Presidential_Results.csv')

# Manipulate counties and calculations
counties = counties[['state_abbr', 'combined_fips', 'votes_dem', 'votes_gop']]
counties['per_dem'] = counties['votes_dem'] / (counties['votes_dem'] + counties['votes_gop'])
counties['per_gop'] = counties['votes_gop'] / (counties['votes_dem'] + counties['votes_gop'])
counties['partisan_masks'] = counties['per_gop'] * (0.74 * 0.6 + (1 - 0.74) * 0.1) + counties['per_dem'] * (0.92 * 0.9 + (1 - 0.92) * 0.3)
del counties['votes_dem']
del counties['votes_gop']
del counties['per_dem']
del counties['per_gop']
# print(counties.groupby('state_abbr').mean())
states = pd.read_csv("masks/mask-prevalence.csv", index_col='State')
# counties[datetime(2020, 5, 23)] = counties.apply(lambda row: row[datetime(2020, 5, 23)] * 0.5 + 0.5 * states.loc[row['state_abbr'], '2020-05-23'] / 100, axis=1)
print(counties)


# Group by State and insert interpolated data between the two raw dates (d1 and d2)
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']
states = df.groupby('State')
df = states.resample('D').mean()
df['percent_wearing_masks'] = df['percent_wearing_masks'].interpolate()


def calculate_linear_equation(x1, y1, x2, y2, x):
    m = (y2 - y1)/(x2 - x1)
    b = y2 - m * x2
    y0 = m * x + b
    return y0


# For each state, extrapolate data
def generate_state_data(s):
    state_name = s.index.values[0][0]
    # Retrieve the date of the first Gathering Restriction date
    d0 = int(interventions['>50 gatherings'].loc[state_name])

    # Calculate mask prevalence on d0 using y = md + b
    d1 = date.toordinal(datetime(2020, 4, 21))  # first poll
    d2 = date.toordinal(datetime(2020, 5, 23))  # second poll
    y1 = s.iloc[0]
    y2 = s.iloc[-1]
    prediction = calculate_linear_equation(d1, y1, d2, y2, d0)

    first_restriction = date.fromordinal(d0)
    first_restriction_dates = pd.date_range(first_restriction, '2020-04-20')
    first_restriction_df = pd.DataFrame(index=first_restriction_dates, columns=['percent_wearing_masks'])
    first_restriction_df.loc[first_restriction] = prediction
    first_restriction_df['State'] = state_name
    first_restriction_df = first_restriction_df.set_index(['State', first_restriction_dates])
    first_restriction_df.index = first_restriction_df.index.rename(['State', 'date'])

    # Extrapolate from d1 to today
    today = date.today()
    most_recent_dates = pd.date_range(datetime(2020, 5, 23), today)
    most_recent_df = pd.DataFrame(index=most_recent_dates, columns=['percent_wearing_masks'])
    most_recent_df.loc[today] = calculate_linear_equation(d1, y1, d2, y2, date.toordinal(today))
    most_recent_df['State'] = state_name
    most_recent_df = most_recent_df.set_index(['State', most_recent_dates])
    most_recent_df.index = most_recent_df.index.rename(['State', 'date'])

    # Interpolate from d0 to today
    state_df = pd.concat([first_restriction_df, s, most_recent_df])
    state_df = state_df.reset_index(level='date')
    state_df['percent_wearing_masks'] = state_df['percent_wearing_masks'].astype('float64').interpolate()

    # Augment mask prevalence from first historical date to first Gathering Restriction
    # as linear from 1% to 5%
    first_historical_date = datetime(2020, 2, 5)
    pre_restriction_dates = pd.date_range(first_historical_date, date.fromordinal(d0-1))
    pre_restriction_df = pd.DataFrame(index=pre_restriction_dates, columns=['percent_wearing_masks'])
    pre_restriction_df.loc[first_historical_date] = 1.0
    pre_restriction_df.loc[date.fromordinal(d0-1)] = 5.0
    pre_restriction_df['State'] = state_name
    pre_restriction_df = pre_restriction_df.set_index(['State', pre_restriction_dates])
    pre_restriction_df.index = pre_restriction_df.index.rename(['State', 'date'])
    pre_restriction_df['percent_wearing_masks'] = pre_restriction_df['percent_wearing_masks'].astype('float64').interpolate()
    pre_restriction_df.reset_index(level='date', inplace=True)

    state_df = pd.concat([pre_restriction_df, state_df])

    state_df['percent_wearing_masks'] = state_df['percent_wearing_masks'] * 0.01
    state_df['reduction_from_baseline'] = state_df['percent_wearing_masks'] * 0.46

    all_states[state_name] = state_df


def generate_county_data(c):

    # Get values from dataframe
    state = c.iloc[0, 0]
    fips = c.iloc[0, 1]
    partisan_masks = c.iloc[0, 2]

    # Convert state dataframe to county dataframe
    df = all_states[state]
    df = df.reset_index()
    df.insert(1, 'fips', fips)
    first_restriction_date = date.fromordinal(int(interventions['>50 gatherings'].loc[state]))
    df['percent_wearing_masks'] = df.apply(lambda row: row['percent_wearing_masks'] * 0.5 + 0.5 * partisan_masks if row['date'] >= first_restriction_date else row['percent_wearing_masks'], axis=1)

    # Update mask efficacy calculation
    df['reduction_from_baseline'] = df['percent_wearing_masks'] * 0.46

    county_dfs.append(df)


all_states = dict()
county_dfs = []
df.groupby('State').apply(generate_state_data)
counties.groupby('combined_fips').apply(generate_county_data)
all_counties_df = pd.concat(county_dfs)
print(all_counties_df)
# all_counties_df.index = all_states_df.index.rename('region')

all_counties_df.to_csv("masks/mask_data_counties_full.csv")
