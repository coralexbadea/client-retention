from fastapi import FastAPI
import pandas as pd
import json
from sklearn.ensemble import RandomForestClassifier
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InputData(BaseModel):
    canceled: str
    client_city: str
    client_country_code: str
    client_gender: str
    client_region: str
    new_customer: str
    owner_name: str
    service_name: str
    
    def to_dict(self):
        return self.dict()
    
    
model = pd.read_pickle('./assets/your_model.pkl')

def analyze_client(client_data):
    global model
    
    X_client = client_data
    # Get feature importances from the trained model
    feature_importances = model.feature_importances_

    # Match feature importances with feature names
    feature_names = X_client.columns
    print("HERE-1",len(feature_importances), len(feature_names))

    # Create a DataFrame to visualize feature importances
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    # Make a prediction for the client
    prediction = model.predict(X_client)[0]

    print("Prediction: ", prediction)
    print(feature_importance_df)

    sorted_feature_importances = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
    feature_importance_dict = {
        'prediction': str(prediction),
        'Feature': [f[0] for f in sorted_feature_importances],
        'Importance': [f[1] for f in sorted_feature_importances]
    }
    feature_importance_json = json.dumps(feature_importance_dict, indent=4)

    return feature_importance_json, feature_importance_dict


def create_and_populate_object(client):
  new_object = pd.read_pickle('./assets/empty_object.pkl')

  for i,j in client.items():
    if(f"{i}_{j}" in new_object.keys()):
      new_object[f"{i}_{j}"] = 1

  return new_object


@app.post("/predict")
def predict(data: InputData):
    client = data.to_dict()
    new_dict = create_and_populate_object(client)
    test_client = pd.DataFrame(new_dict, index=['31326'])
    my_json, my_dict = analyze_client(test_client)
    return my_json



