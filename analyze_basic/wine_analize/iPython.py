import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

df = pd.read_csv('winequality-white.csv', sep=';')
print(df.head())
df.density.hist()
plt.show()
