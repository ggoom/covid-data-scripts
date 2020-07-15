import pandas as pd
from datetime import date
from datetime import datetime
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

VERSION = "_v3"
# Load raw mask data & intervention data & county-level election data
masks = pd.read_csv("masks/covidstates_mask_data{}.csv".format(VERSION), index_col='State')
ipsos = pd.read_csv("masks/ipsos_mask_data{}.csv".format(VERSION), index_col='State')
del ipsos['Unnamed: 0']
masks['2020-06-22'] = ipsos.sort_values(by='State')['2020-06-22'].tolist()
# del masks['Unnamed: 0']
interventions = pd.read_csv("masks/interventions.csv").groupby('STATE').mean(numeric_only=True).round().fillna('737500')
counties = pd.read_csv('masks/2016_US_County_Level_Presidential_Results.csv')

# Manipulate counties and partisan calculations
counties = counties[['state_abbr', 'combined_fips', 'votes_dem', 'votes_gop']]
counties['per_dem'] = counties['votes_dem'] / (counties['votes_dem'] + counties['votes_gop'])
counties['per_gop'] = counties['votes_gop'] / (counties['votes_dem'] + counties['votes_gop'])
counties['partisan_masks'] = 100 * (counties['per_gop'] * (0.74 * 0.6 + (1 - 0.74) * 0.1) + counties['per_dem'] * (0.92 * 0.9 + (1 - 0.92) * 0.3))
del counties['votes_dem']
del counties['votes_gop']
del counties['per_dem']
del counties['per_gop']
# states = pd.read_csv("masks/mask-prevalence.csv", index_col='State')


# Group by State and insert interpolated data between the two raw dates (d1 and d2)
# df['date'] = pd.to_datetime(df['date'])
# df.index = df['date']
# del df['date']
# states = df.groupby('State')
# df = states.resample('D').mean()
# df['percent_wearing_masks'] = df['percent_wearing_masks'].interpolate()


def calculate_linear_equation(x1, y1, x2, y2, x):
    m = (y2 - y1)/(x2 - x1)
    b = y2 - m * x2
    y0 = m * x + b
    return y0


def sigmoid(x, L, x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0)))+b
    return (y)


# For each state, extrapolate data
def generate_county_data(s):
    state_name = s.iloc[0]['state_abbr']
    partisan_masks = s.iloc[0]['partisan_masks']
    print(s.iloc[0])

    # Retrieve the date of the first Gathering Restriction date
    d00 = date.toordinal(datetime(2020, 2, 5))
    d0 = int(interventions['>50 gatherings'].loc[state_name])

    # Calculate mask prevalence on d0 using y = md + b
    d1 = date.toordinal(datetime(2020, 4, 21))  # first poll
    d2 = date.toordinal(datetime(2020, 5, 23))  # second poll
    y1 = midpoint(masks.loc[state_name, '2020-04-21'], partisan_masks)
    y2 = midpoint(masks.loc[state_name, '2020-05-23'], partisan_masks)

    x = [d00, d0, d1, d2]
    y = [1, 5, y1, y2]
    p0 = [60, np.median(x), 0.2, 0.99]
    popt, _ = curve_fit(sigmoid, x, y, p0, bounds=([40, d0, 0.15, 0.9], [100, d1, np.inf, 1]), method='dogbox')

    sigmoid_ordinals = range(d00, d2)
    sigmoid_df = pd.DataFrame({'date': sigmoid_ordinals})
    sigmoid_df['per_masks'] = 0.0
    for row in sigmoid_df.index:
        sigmoid_df['per_masks'][row] = sigmoid(sigmoid_df['date'][row], *popt)

    # Linear fit for later dates
    x = [d2, date.toordinal(datetime(2020, 6, 22))]
    y = [y2, midpoint(masks.loc[state_name, '2020-06-22'], partisan_masks)]
    f = interp1d(x, y, fill_value="extrapolate")

    linear_ordinals = range(d2+1, date.toordinal(date.today()))
    linear_df = pd.DataFrame({'date': linear_ordinals})
    linear_df['per_masks'] = 0.0
    for row in linear_df.index:
        linear_df['per_masks'][row] = f(linear_df['date'][row])

    county_df = pd.concat([sigmoid_df, linear_df])
    county_df.reset_index()
    for row in county_df.index:
        county_df.iloc[row]['date'] = date.fromordinal(int(county_df.iloc[row]['date']))

    # plt.plot([d00, d0, d1] + x, [1, 5, y1] + y, "o", county_df['date'].tolist(), county_df['per_masks'].tolist(), '-')
    # plt.ylabel('per_masks')
    # plt.xlabel('date (ordinal)')
    # plt.show()
    county_df['fips'] = s.iloc[0]['combined_fips']
    county_df['state'] = state_name
    county_dfs.append(county_df)


def midpoint(x1, x2):
    return (x1 + x2) * 0.5


def generate_state_data(s):
    state_name = s.index[0]
    # partisan_masks = s.iloc[0]['partisan_masks']
    print(s.iloc[0])

    # Retrieve the date of the first Gathering Restriction date
    d00 = date.toordinal(datetime(2020, 2, 5))
    d0 = int(interventions['>50 gatherings'].loc[state_name])

    # Calculate mask prevalence on d0 using y = md + b
    d1 = date.toordinal(datetime(2020, 4, 21))  # first poll
    d2 = date.toordinal(datetime(2020, 5, 23))  # second poll
    y1 = masks.loc[state_name, '2020-04-21']
    y2 = masks.loc[state_name, '2020-05-23']

    x = [d00, d0, d1, d2]
    y = [1, 5, y1, y2]
    p0 = [60, np.median(x), 0.2, 0.99]
    popt, _ = curve_fit(sigmoid, x, y, p0, bounds=([40, d0, 0.15, 0.9], [100, d1, np.inf, 1]), method='dogbox')

    sigmoid_ordinals = range(d00, d2)
    sigmoid_df = pd.DataFrame({'date': sigmoid_ordinals})
    sigmoid_df['per_masks'] = 0.0
    for row in sigmoid_df.index:
        sigmoid_df['per_masks'][row] = sigmoid(sigmoid_df['date'][row], *popt)

    # Linear fit for later dates
    x = [d2, date.toordinal(datetime(2020, 6, 22))]
    y = [y2, masks.loc[state_name, '2020-06-22']]
    f = interp1d(x, y, fill_value="extrapolate")

    linear_ordinals = range(d2+1, date.toordinal(date.today()))
    linear_df = pd.DataFrame({'date': linear_ordinals})
    linear_df['per_masks'] = 0.0
    for row in linear_df.index:
        linear_df['per_masks'][row] = f(linear_df['date'][row])

    state_df = pd.concat([sigmoid_df, linear_df])
    state_df.reset_index()
    for row in state_df.index:
        state_df.iloc[row]['date'] = date.fromordinal(int(state_df.iloc[row]['date']))

    # plt.plot([d00, d0, d1] + x, [1, 5, y1] + y, "o", county_df['date'].tolist(), county_df['per_masks'].tolist(), '-')
    # plt.ylabel('per_masks')
    # plt.xlabel('date (ordinal)')
    # plt.show()
    state_df['state'] = state_name
    state_dfs.append(state_df)


county_dfs = []
counties.groupby('combined_fips').apply(generate_county_data)
all_counties_df = pd.concat(county_dfs)
print(all_counties_df)
all_counties_df.to_csv("masks/mask_data_counties_full{}.csv".format(VERSION))

# state_dfs = []
# states.groupby('State').apply(generate_state_data)
# all_states_df = pd.concat(state_dfs)
# print(all_states_df)
# all_states_df.to_csv("masks/mask_data_states_full.csv")
