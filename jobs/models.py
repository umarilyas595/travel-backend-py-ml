import requests
import json
import pandas as pd
from model.recommend import Recommend

def generate_recommend_model():

    res = requests.get('https://tripbackend.dfysaas.com/google')
    data = json.loads(res.text)
    google_data = pd.DataFrame(data)

    # drop extra columns and remove duplicates
    google_data.drop(['createdAt', 'updatedAt'], axis=1, inplace=True)
    # google_data = google_data.dropna(axis=0)

    google_data.rename(columns={'street': 'address'}, inplace=True)
    google_data.dropna(subset=['address'], how='all', inplace=True)
    google_data.drop_duplicates(subset=['address'], inplace=True)

    # reset indexes
    google_data.reset_index(inplace=True)
    google_data.drop(columns=['index'], inplace=True)
    google_data['source'] = 'google'
    google_data['types'] =  [','.join(_type).lower() for _type in google_data['types']]


    # res = requests.get('https://tripbackend.dfysaas.com/location')
    # data = json.loads(res.text)
    # trip_advisor_data = pd.DataFrame(data)

    # # drop extra columns and remove duplicates
    # trip_advisor_data.drop(['postalcode', 'country', 'city', 'createdAt', 'updatedAt'], axis=1, inplace=True)
    # trip_advisor_data = trip_advisor_data.dropna(axis=0)
    # trip_advisor_data.rename(columns={'street': 'address'}, inplace=True)
    # trip_advisor_data.dropna(subset=['address'], how='all', inplace=True)
    # trip_advisor_data.drop_duplicates(subset=['address'], inplace=True)

    # # reset indexes
    # trip_advisor_data.reset_index(inplace=True)
    # trip_advisor_data.drop(columns=['index'], inplace=True)
    # trip_advisor_data['source'] = 'trip-advisor'

    # result = pd.concat([google_data, trip_advisor_data])
    result = google_data
    recommend = Recommend(result)
    recommend.save_by_cosine_similarity('linear-kernal')

    result.to_csv('./model/data.csv', index=False)
    return result

def generate_model():
    print("Generating model ...")

    result = generate_recommend_model()
    
    print("Model generated.")
    return result