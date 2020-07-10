import pandas as pd
import matplotlib.pyplot as plt


def get_state(x):
    try:
        return x.split(' - ')[-2]
    except:
        return 'None'


df = pd.read_csv("infections_timeseries.csv")
# df.loc['Total'] = df.sum()
# df = df.append(sum_df)
df['State'] = df['Combined_Key'].apply(lambda x: get_state(x))
# df = df.transpose()
df = df.groupby(['State']).sum()
df.drop('FIPS')
row = df.loc['Massachusetts']
row.plot()
plt.show()


# df.to_csv("infections_timeseries_edited.csv", index=False)
