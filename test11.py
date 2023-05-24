import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("ggplot")


filename = f'2023-05-13-run12.csv'

df = pd.read_csv(filename)

fig = plt.figure(figsize=(10, 6))
fig.suptitle('interp')

ax = fig.add_subplot(111)
ax.plot(df['mso_timestamp_t'][0:6], df['mso_area_t'][0:6], label='mso_area', c='blue')
ax.scatter(df['mso_timestamp_t'][0:6], df['mso_area_t'][0:6], label='mso_area', c='blue')

ax.plot(df['gmd1_timestamp_t'][0:5], df['gmd1_t'][0:5]*700, label='gmd1', c='green')
ax.plot(df['gmd1_timestamp_t'][0:5]+0.2, df['gmd1_t'][0:5]*700, label='gmd1_delta',linestyle='--', c='green')
ax.scatter(df['gmd1_timestamp_t'][0:5]+0.2, df['gmd1_t'][0:5]*700, label='gmd1_delta', c='green')

mso_area_interp = np.interp(df['gmd1_timestamp_t'][0:5]+0.2, df['mso_timestamp_t'][0:6], df['mso_area_t'][0:6])
ax.scatter(df['gmd1_timestamp_t'][0:5]+0.2, mso_area_interp, label='mso_area_interp', c='red')

# ax.legend(loc='upper right')
plt.show()