import json

def JSONResponse(df):
    result = df.to_dict(orient="records")
    result = json.dumps(result)
    result = json.loads(result)
    return result