import json
from process_features import *
import pandas
import datetime
import operator
from collections import Counter
from functools import reduce

# load train and test datasets
data = json.loads(open("Data/train.json").read())
test = json.loads(open("Data/test.json").read())

# preprocess & construct training dataset
created = dict().fromkeys(data["created"])
features = dict().fromkeys(data["features"])
interest_level = dict().fromkeys(data["interest_level"])
addresses = dict().fromkeys(data["display_address"])
st_addresses = dict().fromkeys(data["street_address"])
photos = dict().fromkeys(data["photos"])

for i in created.keys():
    created[i] = (datetime.datetime.now() - datetime.datetime.strptime(data["created"][i], "%Y-%m-%d %H:%M:%S")).days

for i in created.keys():
    features[i] = preprocess_features(data["features"][i])
    addresses[i] = ' '.join(data["display_address"][i].lower().split(' '))
    st_addresses[i] = ' '.join(data["street_address"][i].lower().split(' '))
    photos[i] = len(data["photos"][i])

features = split_feature(features)

samples = pandas.DataFrame({"bathrooms": data["bathrooms"], "bedrooms": data["bedrooms"],
                            "building_id": data["building_id"], "created": created,
                            "description": data["description"], "display_address": addresses,
                            "features": features, "latitude": data["latitude"],
                            "listing_id": data["listing_id"], "longitude": data["longitude"],
                            "manager_id": data["manager_id"], "photos": data["photos"], "price": data["price"],
                            "street_address": st_addresses, "interest_level": data["interest_level"],
                            "photos_num": photos})
print((set(samples["photos_num"])), len(samples))


