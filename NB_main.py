###########################
#Author Yili Zou
#Naive Bayes Final Project
###########################

import json
import math
from numpy.random import choice
from sklearn.metrics import log_loss
# def convert_keys_to_string(dictionary):
#     """Recursively converts dictionary keys to strings."""
#     if not isinstance(dictionary, dict):
#         return dictionary
#     return dict((str(k), convert_keys_to_string(v))
#         for k, v in dictionary.items())

#get photo count of the photo data feature
def get_photo_count(df_photos):
    df_temp = df_photos
    for x in df_temp:
        df_temp[x] = len(df_temp[x])
    return df_temp

#get number of apartment features
def get_apt_feature_number(df_features):
    df_temp = df_features
    for x in df_temp:
        df_temp[x] = len(df_temp[x])
    return df_temp


#convert the price to diecrete values, threshold is 100
def price_discretization(df_price):
    df_temp = df_price
    for x in df_temp:
        df_temp[x]=int((df_price[x])/300)
    return df_temp

#prepare the data for latitude, convert it to discrete number
def latitude_discretization(df_latitude):
    df_temp = df_latitude
    for x in df_temp:
        df_temp[x]=int((df_temp[x]-40)*10000/400)
    return df_temp

#prepare the data for latitude, convert it to discrete number
def longitude_discretization(df_longitude):
    df_temp = df_longitude
    for x in df_temp:
        df_temp[x]=int((df_temp[x]+74)*10000/400)
    return df_temp

#get element count in a dictionary
def dic_count(dic, element):
    count=0
    for x in dic:
        if(dic.get(x)==element):
            count+=1
    return count

def extract_listing_id_by_label(df_label, label):
    iterator=0
    listing_id={}
    for x in df_label:
        if(df_label[x]==label):
            listing_id[iterator]=x
            iterator+=1
    return listing_id

#return the distinct value of the data feature
def get_enumeration(df):
    enumeration= set(df.values())
    return enumeration

def get_sub_data_feature(data_feature, label_id):
    sub_df={}
    for x in label_id:
        sub_df[label_id[x]]=data_feature[label_id[x]]
    return sub_df

def get_conditional_pro(data_feature,label_id):
    condition_pro={}
    data_feature_values = data_feature.values()
    enumeration = set(data_feature_values)
    sub_df={}
    for x in label_id:
        sub_df[label_id[x]]=data_feature[label_id[x]]
    for enum in enumeration:
        count = dic_count(sub_df,enum)
        condition_pro[enum]=(count+1)/(len(label_id)+len(enumeration))

    condition_pro["default"]=1/(len(label_id)+len(enumeration))
    return condition_pro
#get conditional probability dictionary from the conditional probability dictionary, value is the value of the feature
def get_conditional_probability(conditional_pro, value):
    #conditional probability with label, high, medium, and low
    conditional_pro_label={}
    if(value in conditional_pro["high"]):
        conditional_pro_label["high"]=conditional_pro["high"][value]
        conditional_pro_label["medium"] = conditional_pro["medium"][value]
        conditional_pro_label["low"] = conditional_pro["low"][value]
    else:
        conditional_pro_label["high"] = conditional_pro["high"]["default"]
        conditional_pro_label["medium"] = conditional_pro["medium"]["default"]
        conditional_pro_label["low"] = conditional_pro["low"]["default"]
    return conditional_pro_label


def predict_label(dataset,conditional_pro,priors):
    result={}
    probabilities = {}

    #get data features
    df_bathrooms = dataset["bathrooms"]
    df_bedrooms = dataset["bedrooms"]
    df_prices = price_discretization(dataset["price"])
    df_photos = get_photo_count(dataset["photos"])
    df_latitude = latitude_discretization(dataset["latitude"])
    df_longitude = longitude_discretization(dataset["longitude"])
    df_features = get_apt_feature_number(dataset["features"])

    #get data feature conditional probabilities
    bathrooms_conditional_pro=conditional_pro["bathrooms"]
    bedrooms_conditional_pro = conditional_pro["bedrooms"]
    prices_conditional_pro = conditional_pro["price"]
    photos_conditional_pro = conditional_pro["photos"]
    latitude_conditional_pro = conditional_pro["latitude"]
    longitude_conditional_pro = conditional_pro["longitude"]
    features_conditional_pro = conditional_pro["features"]

    #for each apartment listing
    for x in df_bathrooms:
        #get feature values
        bathroom_value=df_bathrooms[x]
        bedroome_value=df_bedrooms[x]
        price_value=df_prices[x]
        photo_value = df_photos[x]
        latitude_value = df_latitude[x]
        longitude_value = df_longitude[x]
        features_value = df_features[x]



        #get conditional probability with labels
        conditional_pro_label_bathroom = get_conditional_probability(bathrooms_conditional_pro,bathroom_value)
        conditional_pro_label_bedroom = get_conditional_probability(bedrooms_conditional_pro,bedroome_value)
        conditional_pro_label_price = get_conditional_probability(prices_conditional_pro,price_value)
        conditional_pro_label_photo = get_conditional_probability(photos_conditional_pro, photo_value)
        conditional_pro_label_latitude = get_conditional_probability(latitude_conditional_pro, latitude_value)
        conditional_pro_label_longitude = get_conditional_probability(longitude_conditional_pro, longitude_value)
        conditional_pro_label_features = get_conditional_probability(features_conditional_pro, features_value)

        #calculate score
        score_high=math.log10(priors["high"])+conditional_pro_label_bathroom["high"]+\
                     conditional_pro_label_bedroom["high"]+conditional_pro_label_price["high"]+conditional_pro_label_photo["high"]\
                   + conditional_pro_label_latitude["high"]+\
                     conditional_pro_label_longitude["high"]+conditional_pro_label_features["high"]
        score_medium = math.log10(priors["medium"]) + conditional_pro_label_bathroom["medium"] + \
                     conditional_pro_label_bedroom["medium"] + conditional_pro_label_price["medium"]+conditional_pro_label_photo["medium"]\
                       + conditional_pro_label_latitude["medium"]+ \
                     conditional_pro_label_longitude["medium"]+conditional_pro_label_features["medium"]
        score_low = math.log10(priors["low"]) + conditional_pro_label_bathroom["low"] + \
                     conditional_pro_label_bedroom["low"] + conditional_pro_label_price["low"]+conditional_pro_label_photo["low"]\
                    + conditional_pro_label_latitude["medium"]+ \
                     conditional_pro_label_longitude["low"]+conditional_pro_label_features["low"]
        temp_pro_high= math.pow(score_high,10)
        temp_pro_medium = math.pow(score_medium, 10)
        temp_pro_low = math.pow(score_low, 10)
        temp_total_pro=temp_pro_high+temp_pro_medium+temp_pro_low

        
        pro_high=temp_pro_high/temp_total_pro
        pro_medium=temp_pro_medium/temp_total_pro
        pro_low=temp_pro_low/temp_total_pro


        choice_arr = ['high', 'medium', 'low']
        prediction = choice(choice_arr, 1, p=[pro_high, pro_medium, pro_low])
        result[x]=prediction[0]
        #dictionary for listing id as key, and probabilities as value
        probabilities[x]=[pro_high,pro_medium,pro_low]

    return result, probabilities

with open('train_1.json') as data_file:
    data = json.load(data_file)
#get data features needed
df_bathrooms = data["bathrooms"]
df_bedrooms = data["bedrooms"]
df_price = price_discretization(data["price"])
df_photos = get_photo_count(data["photos"])
df_latitude = latitude_discretization(data["latitude"])
df_longitude = longitude_discretization(data["longitude"])
df_features = get_apt_feature_number(data["features"])


low_id = extract_listing_id_by_label(data["interest_level"],"low")
medium_id = extract_listing_id_by_label(data["interest_level"],"medium")
high_id = extract_listing_id_by_label(data["interest_level"],"high")

#get count for each label
count_low = len(low_id)
count_medium = len(medium_id)
count_high = len(high_id)
total_count = count_low +count_medium+count_high
#get priors for each label
priors={}
priors["low"] = float(count_low/total_count)
priors["medium"] = float(count_medium/total_count)
priors["high"] = float(count_high/total_count)


#calculate conditional probabilities for features
#bathroom feature
bathrooms_conditional_pro={}
bathrooms_conditional_pro["high"] = get_conditional_pro(df_bathrooms,high_id)
bathrooms_conditional_pro["medium"] = get_conditional_pro(df_bathrooms,medium_id)
bathrooms_conditional_pro["low"] = get_conditional_pro(df_bathrooms,low_id)
#bedroom feature
bedrooms_conditional_pro={}
bedrooms_conditional_pro["high"] = get_conditional_pro(df_bedrooms,high_id)
bedrooms_conditional_pro["medium"] = get_conditional_pro(df_bedrooms,medium_id)
bedrooms_conditional_pro["low"] = get_conditional_pro(df_bedrooms,low_id)
#price feature
price_conditional_pro={}
price_conditional_pro["high"] = get_conditional_pro(df_price,high_id)
price_conditional_pro["medium"] = get_conditional_pro(df_price,medium_id)
price_conditional_pro["low"] = get_conditional_pro(df_price,low_id)
#number of photos feature
photos_conditional_pro={}
photos_conditional_pro["high"] = get_conditional_pro(df_photos,high_id)
photos_conditional_pro["medium"] = get_conditional_pro(df_photos,medium_id)
photos_conditional_pro["low"] = get_conditional_pro(df_photos,low_id)
#latitude data feature
latitude_conditional_pro={}
latitude_conditional_pro["high"] = get_conditional_pro(df_latitude,high_id)
latitude_conditional_pro["medium"] = get_conditional_pro(df_latitude,medium_id)
latitude_conditional_pro["low"] = get_conditional_pro(df_latitude,low_id)
#longitude data feature
longitude_conditional_pro={}
longitude_conditional_pro["high"] = get_conditional_pro(df_longitude,high_id)
longitude_conditional_pro["medium"] = get_conditional_pro(df_longitude,medium_id)
longitude_conditional_pro["low"] = get_conditional_pro(df_longitude,low_id)
#apartment features data feature
features_conditional_pro={}
features_conditional_pro["high"] = get_conditional_pro(df_features,high_id)
features_conditional_pro["medium"] = get_conditional_pro(df_features,medium_id)
features_conditional_pro["low"] = get_conditional_pro(df_features,low_id)

#combine conditonal probabilities
conditional_pro={}
conditional_pro["bathrooms"]=bathrooms_conditional_pro
conditional_pro["bedrooms"]=bedrooms_conditional_pro
conditional_pro["price"]=price_conditional_pro
conditional_pro["photos"]=photos_conditional_pro
conditional_pro["latitude"]=latitude_conditional_pro
conditional_pro["longitude"]=longitude_conditional_pro
conditional_pro["features"]=features_conditional_pro

print("Training finished. Now testing for accuracy on another dataset...")
with open('train_2.json') as data_file:
    testing_data = json.load(data_file)
labels = testing_data["interest_level"]
result = predict_label(testing_data,conditional_pro,priors)
prediction=result[0]
correct_count = 0
for x in prediction:
    if(labels[x]==prediction[x]):
        correct_count+=1

# list_prediction=[]
# list_label=[]
# for x in labels:
#     list_label.append(labels[x])
# for x in prediction:
#     list_prediction.append(prediction[x])
# print(log_loss(list_label,list_prediction))



print("The accuracy is : ", float(correct_count/len(prediction)))
probabilities = result[1]
f = open('result.txt', 'w')
f.write('listing id          '+'high probability         '+'medium probability         '+'low probability\n')
for x in probabilities:
    f.write(x+'              ')
    for y in probabilities[x]:
        f.write(str(y)+'      ')
    f.write('\n')


