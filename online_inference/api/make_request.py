import random

import numpy as np
import pandas as pd
import requests
from api.app import TOO_MUCH_DATA_CONSTRAINT

if __name__ == "__main__":
    data = pd.read_csv("data/raw/heart.csv")
    data = data.drop("target", axis=1)
    request_features = list(data.columns)
    print(request_features)
    for i in range(100):
        request_data = []
        for j in range(random.randint(1, TOO_MUCH_DATA_CONSTRAINT + 5)):
            request_data.append([
                x.item() if isinstance(x, np.generic) else x for x in data.iloc[i].tolist()
            ])
        print(request_data)
        response = requests.get(
            "http://127.0.0.1:5000/predict/",
            json={"data": request_data, "features": request_features},
        )
        print(response.status_code)
        print(response.json())
