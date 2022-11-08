import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('Data for Predicting N.csv')

s5 = df['S5']
census_value = df['N']

# print(census_value)
model = lr().fit(s5.values.reshape(-1,1), df['N'])
r_sq = model.score(s5.values.reshape(-1,1), census_value)
# print(r_sq)

# print(df)

# want to pivot data longer so that there's a type column
# with predictor types (housing, res, s5, s10, n) and year
# so there should be a bunch of values for 2015 entries, etc
df_long = pd.melt(df, id_vars='Year',
        value_vars=['Housing', 'Reservations', 'S5', 'S10', 'N'])

df_wide = pd.pivot()

sns.lineplot(data=df_long, x='Year', y='value', hue='variable')
plt.show()