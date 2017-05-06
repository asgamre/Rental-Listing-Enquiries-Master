import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
#this code  to show the Listing_ID and the probility of high medium and low
df = pd.read_csv(open("submission_rf.csv", "r"))
print(df.head(10))
#myplot = df.plot(kind='bar')
#plt.show()
print(df.shape)
df.describe()
df_ID = df.set_index("listing_id")
df_ID[:10].plot(kind='bar')
plt.title('Listing_ID and probility of high medium and low')
plt.show()
'''

'''
#this code  to show the price and the probility of high medium and low
df = pd.read_json(open("train.json", "r"))
print(df.head(10))
#myplot = df.plot(kind='bar')
#plt.show()
print(df.shape)
df.describe()
df_ID = df.set_index("listing_id")
df_price = df.set_index("price")
df_ID[:50].plot(kind='bar')
plt.title("Listing_ID and it's features" )
plt.show()
'''