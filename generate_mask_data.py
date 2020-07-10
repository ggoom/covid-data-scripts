import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np

# Load raw mask data & intervention data
df = pd.read_csv("masks/mask-prevalence.csv", index_col=0)
df = pd.melt(df, id_vars=['State'], value_vars=['2020-04-21', '2020-05-23'])
df = df.rename(columns={'variable': 'date', 'value': 'percent_wearing_masks'})
interventions = pd.read_csv("masks/interventions.csv").groupby('STATE').mean(numeric_only=True).round().fillna('737500')

# Group by State and insert interpolated data between the two raw dates (d1 and d2)
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']
states = df.groupby('State')
df = states.resample('D').mean()
df['percent_wearing_masks'] = df['percent_wearing_masks'].interpolate()

# For each state, extrapolate data to the date of the first Gathering Restriction (d0)
all_states = []


def calculate_linear_equation(x1, y1, x2, y2, x):
    m = (y2 - y1)/(x2 - x1)
    b = y2 - m * x2
    y0 = m * x + b
    return y0


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

    all_states.append(state_df)


df.groupby('State').apply(generate_state_data)
all_states_df = pd.concat(all_states)
all_states_df.index = all_states_df.index.rename('region')

print(all_states_df)
all_states_df.to_csv("masks/mask_data_full.csv")
