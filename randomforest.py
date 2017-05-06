import numpy as np 
import pandas as pd 

from forest import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

df = pd.read_json(open("train.json", "r"))
#print(df.shape)
#print(df.head())

df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day

num_feats = ["bathrooms", "bedrooms", "latitude", "longitude", "price",
             "num_photos", "num_features", "num_description_words",
             "created_year", "created_month", "created_day"]
X = df[num_feats]
y = df["interest_level"]
print(X.head(n=5))
#print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

clf = RandomForestClassifier(n_estimators=1000)
clf.fit(X_train, y_train)
y_test_pred = clf.predict_proba(X_test)
print("the log loss is: ")
print(log_loss(y_test, y_test_pred))


df = pd.read_json(open("test.json", "r"))
# print(df.shape)
df["num_photos"] = df["photos"].apply(len)
df["num_features"] = df["features"].apply(len)
df["num_description_words"] = df["description"].apply(lambda x: len(x.split(" ")))
df["created"] = pd.to_datetime(df["created"])
df["created_year"] = df["created"].dt.year
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
X = df[num_feats]

y = clf.predict_proba(X)

labels2idx = {label: i for i, label in enumerate(clf.classes_)}


sub = pd.DataFrame()
sub["listing_id"] = df["listing_id"]
for label in ["high", "medium", "low"]:
    sub[label] = y[:, labels2idx[label]]
sub.to_csv("submission_rf.csv", index=False)