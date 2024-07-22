from fastapi import FastAPI, Response
import pandas as pd
from pydantic import BaseModel
from model.recommend import Recommend
import json
from fastapi.middleware.cors import CORSMiddleware
from jobs.models import generate_recommend_model
import sys
import re
sys.path.append('.')
from jobs.my_jobs import scheduler
from model.functions import JSONResponse
import requests

app = FastAPI()

origins = [
    "https://weplan.ai",
    "http://weplan.ai",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class similarity(BaseModel):
    id: str

class recommendProps(BaseModel):
    input: str
    types: str

scheduler.start()

@app.get('/test-google')
async def test_google():
    res = requests.get('https://tripbackend.dfysaas.com/google')
    data = json.loads(res.text)
    df = pd.DataFrame(data)
    print(df)
    return data

@app.get('/save-model')
async def get_model():

    result = generate_recommend_model()
    result = result.fillna('')

    result = result.to_dict(orient="records")
    result = json.dumps(result)
    result = json.loads(result)

    return result

@app.post('/get-recommendation')
async def getRecommendations(data : recommendProps):

    df = pd.read_csv('./model/data.csv')
    address_input = data.input

    pattern = r'\b[a-zA-Z]*\d+(?:-\d+)?, \b'
    address_input = re.sub(pattern, '', address_input)

    # define a function to search string from array
    def search_string(s, search):
        return str(search).lower() in str(s).lower()

    filter_df = pd.DataFrame({})

    # if input length is 1 word then data can be from this location
    # if len(address_input.split(" ")) <= 2:
    #     newdf = df.applymap(lambda x: search_string(x, address_input))
    #     filter_df = df.loc[newdf['address']]
        

    # if result found from 1 word of input then display result without recommendation algorithm
    if filter_df.shape[0] > 2 and len(address_input.split(" ")) == 1 and data.types :
        filter_df.fillna('', inplace=True)

        # if types exist then filter results from multiple types
        if data.types:
            recommend = Recommend(filter_df)
            similar_documents = recommend.get_recommendations_by_name(data.types, 30)
            filter_df = df.iloc[similar_documents]

        result = JSONResponse(filter_df)
        return {'recommendations': result}

    recommend = Recommend(df)
    similar_documents = recommend.get_recommendations(address_input, 30)
    
    # if types are available then filter data by types
    if len(data.types) > 0:
        similar_documents = recommend.filter_by_types(data.types.split(','))

    result = similar_documents.to_dict(orient="records")
    result = json.dumps(result)
    result = json.loads(result)
    
    return {'recommendations': result}
