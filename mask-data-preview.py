import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("masks/mask_data_counties_full.csv")
df = df[df['fips'] == 25025]
plt.plot(df['per_masks'].tolist())
print(df)
