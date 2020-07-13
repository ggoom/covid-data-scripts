import pandas as pd

us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands': 'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}


def calc_mask_score(s):
    always = int(s.iloc[0][:-1])
    sometimes = int(s.iloc[1][:-1])
    occasionally = int(s.iloc[2][:-1])
    # never = int(s.iloc[3][:-1])
    return round(always * 1 + sometimes * 0.6 + occasionally * 0.3, 1)


df = pd.read_csv('/Users/kjjin/TheBridge/covid/masks/AxiosIpsos_masks_states.txt', sep='\t', header=0)
regions = df.groupby('GEO ').agg({'May 8-June 22': calc_mask_score})
regions = regions.rename({'GEO ': 'State', 'May 8-June 22': '2020-06-22'})

for state in ['Alaska', 'Washington', 'Hawaii']:
    regions.loc[state] = regions.loc['Pacific']

for state in ['Montana', 'Wyoming', 'Idaho', 'New Mexico']:
    regions.loc[state] = regions.loc['Mountain']

for state in ['North Dakota', 'South Dakota', 'Nebraska']:
    regions.loc[state] = regions.loc['West North Central']

for state in ['West Virginia', 'Delaware', 'District of Columbia']:
    regions.loc[state] = regions.loc['South Atlantic']

for state in ['Maine', 'Vermont', 'New Hampshire', 'Rhode Island']:
    regions.loc[state] = regions.loc['New England']


regions = regions.reset_index()
regions = regions.rename(columns={'GEO ': 'State', 'May 8-June 22': '2020-06-22'})
states = regions[[True if region in us_state_abbrev else False for region in regions['State']]]

index_list = states['State'].tolist()
new_indices = []
for index in index_list:
    if index in us_state_abbrev:
        new_indices.append(us_state_abbrev[index])
states['State'] = new_indices

states.to_csv("masks/ipsos_mask_data.csv")
