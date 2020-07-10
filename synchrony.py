import scipy.stats as stats
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# state = "MA"
county = 25009

mobility = pd.read_csv("Global_Mobility_Report.csv")
mobility = mobility[mobility['census_fips_code'] == county]
mobility.index = mobility['date']
mobility['mobility'] = mobility['retail_and_recreation_percent_change_from_baseline'] + \
    mobility['grocery_and_pharmacy_percent_change_from_baseline'] + \
    mobility['transit_stations_percent_change_from_baseline'] + \
    mobility['workplaces_percent_change_from_baseline']
mobility = mobility[['mobility']].copy()
# print(mobility)
# mobility.plot()
# plt.show()


# masks = pd.read_csv(state + "_mask_data.csv")
# masks.rename(columns={'Unnamed: 0': 'date', 'reduction_from_baseline': 'masks'}, inplace=True)
# masks.index = masks['date']
# del masks['date']
# del masks['percent_wearing_masks']
# print(masks)
# masks.plot()
# plt.show()

# rt = pd.read_csv("rt_full_2020_06_19.csv")
# rt = rt[rt['region'] == state]
# rt.index = rt['date']
# rt = rt[['mean']].copy()
# rt.rename(columns={'mean': 'Rt'}, inplace=True)
# print(rt)

cases = pd.read_csv("new_cases_MA_counties_2020_06_28.csv")
cases = cases[cases['fips'] == county]
cases.index = cases['date']
cases = cases[['new_cases']].copy()
print(cases)

# df = mobility.merge(rt, on='date')
# print(df)
# # combined.plot()
# # plt.show()

# overall_pearson_r = df.corr().iloc[0, 1]
# print(f"Pandas computed Pearson r: {overall_pearson_r}")
# # out: Pandas computed Pearson r: 0.2058774513561943

# r, p = stats.pearsonr(df.dropna()['mobility'], df.dropna()['Rt'])
# print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# # out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

# # # Compute rolling window synchrony
# # f, ax = plt.subplots(figsize=(7, 3))
# # df.rolling(window=30, center=True).median().plot(ax=ax)
# # ax.set(xlabel='Time', ylabel='Pearson r')
# # ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")


# # Set window size to compute moving window synchrony.
# r_window_size = 30
# # Interpolate missing data.
# df_interpolated = df.interpolate()
# # Compute rolling window synchrony
# rolling_r = df_interpolated['mobility'].rolling(window=r_window_size, center=True).corr(df_interpolated['Rt'])
# f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
# # df['mobility'].plot(ax=ax[0], title='mobility')
# # df['Rt'].plot(secondary_y=True, ax=ax[0], title='Rt')
# df['mobility'].rolling(window=r_window_size, center=True).median().plot(ax=ax[0], title='mobility')
# df['Rt'].rolling(window=r_window_size, center=True).median().plot(secondary_y=True, ax=ax[0], title='Rt')

# ax[0].set(xlabel='Frame', ylabel='mobility')
# rolling_r.plot(ax=ax[1])
# ax[1].set(xlabel='Frame', ylabel='Pearson r')
# plt.suptitle("data and rolling window correlation for " + state)

# plt.show()

df = mobility.merge(cases, on='date')
print(df)
# combined.plot()
# plt.show()

overall_pearson_r = df.corr().iloc[0, 1]
print(f"Pandas computed Pearson r: {overall_pearson_r}")
# out: Pandas computed Pearson r: 0.2058774513561943

r, p = stats.pearsonr(df.dropna()['mobility'], df.dropna()['new_cases'])
print(f"Scipy computed Pearson r: {r} and p-value: {p}")
# out: Scipy computed Pearson r: 0.20587745135619354 and p-value: 3.7902989479463397e-51

# # Compute rolling window synchrony
# f, ax = plt.subplots(figsize=(7, 3))
# df.rolling(window=30, center=True).median().plot(ax=ax)
# ax.set(xlabel='Time', ylabel='Pearson r')
# ax.set(title=f"Overall Pearson r = {np.round(overall_pearson_r,2)}")


# Set window size to compute moving window synchrony.
r_window_size = 10
# Interpolate missing data.
df_interpolated = df.interpolate()
# Compute rolling window synchrony
rolling_r = df_interpolated['mobility'].rolling(window=r_window_size, center=True).corr(df_interpolated['new_cases'])
f, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
# df['mobility'].plot(ax=ax[0], title='mobility')
# df['new_cases'].plot(secondary_y=True, ax=ax[0], title='Rt')
df['mobility'].rolling(window=r_window_size, center=True).median().plot(ax=ax[0], title='mobility')
df['new_cases'].rolling(window=r_window_size, center=True).median().plot(secondary_y=True, ax=ax[0], title='new_cases')

ax[0].set(xlabel='Frame', ylabel='mobility')
rolling_r.plot(ax=ax[1])
ax[1].set(xlabel='Frame', ylabel='Pearson r')
plt.suptitle("data and rolling window correlation for " + str(county))

plt.show()
