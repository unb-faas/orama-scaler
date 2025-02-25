import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize

df = pd.DataFrame({
    'coluna_1': [10, 12, 14, 15, 16, 18, 19, 100, 105, 110],
    'coluna_2': [5, 7, 9, 10, 12, 13, 50, 60, 70, 80],
    'categoria': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
})

df_winsorizado = df.copy()
df_winsorizado[df_winsorizado.select_dtypes(include=['number']).columns] = \
    df_winsorizado.select_dtypes(include=['number']).apply(lambda x: winsorize(x.to_numpy(), limits=(0.1, 0.1)))

print(df)

print(df_winsorizado)