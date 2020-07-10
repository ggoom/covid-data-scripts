import pandas as pd
import matplotlib.pyplot as plt

mobility = pd.read_csv("Global_Mobility_Report.csv")
mobility = mobility[mobility['iso_3166_2_code'] == "US-MA"]
mobility.index = mobility['date']
mobility['mobility'] = mobility['retail_and_recreation_percent_change_from_baseline'] + \
    mobility['grocery_and_pharmacy_percent_change_from_baseline'] + \
    mobility['parks_percent_change_from_baseline'] + \
    mobility['transit_stations_percent_change_from_baseline'] + \
    mobility['workplaces_percent_change_from_baseline'] + \
    mobility['residential_percent_change_from_baseline']
mobility = mobility[['mobility']].copy()
# print(mobility)
# mobility.plot()
# plt.show()


masks = pd.read_csv("MA_mask_data.csv")
masks.rename(columns={'Unnamed: 0': 'date', 'reduction_from_baseline': 'masks'}, inplace=True)
masks.index = masks['date']
del masks['date']
del masks['percent_wearing_masks']
# print(masks)
# masks.plot()
# plt.show()

rt = pd.read_csv("rt_full_2020_06_19.csv")
rt = rt[rt['region'] == 'MA']
rt.index = rt['date']
rt = rt[['mean']].copy()
rt.rename(columns={'mean': 'Rt'}, inplace=True)
# print(rt)

combined = mobility.merge(rt, on='date')
print(combined)
# combined.plot()
# plt.show()
