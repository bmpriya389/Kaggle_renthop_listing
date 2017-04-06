import json
import datetime
import pandas
import numpy as np
from collections import Counter
from process_features import feature_process
from matplotlib import pyplot as plt

data = json.loads(open("Data/train.json").read())
test = json.loads(open("Data/test.json").read())

created = dict().fromkeys(data["created"])
features = dict().fromkeys(data["features"])
interest_level = dict().fromkeys(data["interest_level"])

features_total = []

for i in created.keys():
    created[i] = (datetime.datetime.now() - datetime.datetime.strptime(data["created"][i], "%Y-%m-%d %H:%M:%S")).days
    features_total.extend(map(lambda x: x.lower(), data["features"][i]))


samples = pandas.DataFrame({"bathrooms": data["bathrooms"], "bedrooms": data["bedrooms"],
                            "building_id": data["building_id"], "created": created,
                            "description": data["description"], "display_address": data["display_address"],
                            "features": data["features"], "latitude": data["latitude"],
                            "listing_id": data["listing_id"], "longitude": data["longitude"],
                            "manager_id": data["manager_id"], "photos": data["photos"], "price": data["price"],
                            "street_address": data["street_address"], "interest_level": data["interest_level"]})

sample_low = samples.loc[(samples.interest_level == 'low')]
sample_medium = samples.loc[(samples.interest_level == 'medium')]
sample_high = samples.loc[(samples.interest_level == 'high')]

feat_dict = Counter(features_total)

feat_low_total = [i.lower() for _ in sample_low["features"] for i in _]
feature_low_dict = Counter(feat_low_total)
feat_medium_total = [i.lower() for _ in sample_medium["features"] for i in _]
feature_medium_dict = Counter(feat_medium_total)
feat_high_total = [i.lower() for _ in sample_high["features"] for i in _]
feature_high_dict = Counter(feat_high_total)

posterior_low = dict.fromkeys(samples.index,0)
posterior_medium = dict.fromkeys(samples.index,0)
posterior_high = dict.fromkeys(samples.index,0)

smoothing = 1
#feat_dict = feature_process(feat_dict)

for i in sample_low.index:
    posterior_low[i] = np.prod([(feature_low_dict[i] + smoothing)/(len(sample_low)) for i in
                                sample_low["features"][i]])

for i in sample_medium.index:
    posterior_medium[i] = np.prod([(feature_medium_dict[i] + smoothing)/(len(sample_medium))) for
                                   i in sample_medium["features"][i]])

for i in sample_high.index:
    posterior_high[i] = np.prod([(feature_high_dict[i] + smoothing)/(len(sample_high) + len(feat_dict.keys())) for i in
                                 sample_high["features"][i]])

posterior = {**posterior_high, **posterior_medium, **posterior_low}
samples["posterior"] = pandas.Series(posterior)

color_set = {"high": "green", "medium": "blue", "low": "red"}
colors = [color_set[k] for k in samples["interest_level"]]
"""
print(Counter(samples["interest_level"]))
plt.scatter(samples["posterior"], samples["price"], c=colors, marker='x')
plt.show()
"""


smoothing = 1

for i in sample_high.index:
    score_high[i] = reduce(lambda x,y: x * y, [term_freq_high[k]+1 for k in sample_high["features"][i]], 1)/len(feat_dict)
    #point_high[i] = reduce(lambda x,y: x + y, [(feature_high_dict[k]['points'] + smoothing) for k in sample_high["features"][i]], 1)
score_high = pandas.Series(score_high)
#point_high = pandas.Series(point_high)

for i in sample_low.index:
    score_low[i] = reduce(lambda x, y: x * y, [term_freq_low[k]+1 for k in sample_low["features"][i]], 1)/ len(feat_dict)
    #point_low[i] = reduce(lambda x, y: x + y, [(feature_low_dict[k]['points'] + smoothing) for k in sample_low["features"][i]], 1)
score_low = pandas.Series(score_low)
#point_low = pandas.Series(point_low)

for i in sample_medium.index:
    score_medium[i] = reduce(lambda x,y: x * y, [term_freq_medium[k]+1 for k in sample_medium["features"][i]], 1)/ len(feat_dict)
    #point_medium[i] = reduce(lambda x,y: x + y, [(feature_medium_dict[k]['points'] + smoothing) for k in sample_medium["features"][i]], 1)
score_medium = pandas.Series(score_medium)
#point_medium = pandas.Series(point_medium)

print(min(score_low), max(score_low))
print(min(point_low), max(point_low))

print(min(score_medium), max(score_medium))
print(min(point_medium), max(point_medium))

print(min(score_high), max(score_high))
print(min(point_high), max(point_high))

score = {**score_high, **score_medium, **score_low}
points = {**point_high, **point_medium, **point_low}

samples["score"] = pandas.Series(score)
samples["points"] = pandas.Series(points)

samples_stat = samples.groupby('interest_level')
print(samples_stat["score"].describe())
print(samples_stat["points"].describe())
print(samples_stat["price"].describe())

print(Counter(samples["interest_level"]))

sample_low = samples.loc[(samples.interest_level == 'low')]
sample_medium = samples.loc[(samples.interest_level == 'medium')]
sample_high = samples.loc[(samples.interest_level == 'high')]

sample_low_medium =([sample_low, sample_medium])
sample_medium_high = pandas.concat([sample_high, sample_medium])
sample_high_low = pandas.concat([sample_low, sample_high])
sample_mod_medium_high = pandas.DataFrame()
sample_mod_medium_high = pandas.DataFrame({'latlonpricescore' : sample_medium_high['score'] + sample_medium_high['points'], 'price': sample_medium_high['price'], 'interest_level': sample_medium_high['interest_level'] })

print(sample_medium.head())
print(sample_high.head())
#sns.pairplot(sample_medium_high, markers='x', hue='interest_level', palette={'high': 'purple', 'medium': 'orange', 'low': 'red'}, size = 1)

sns.pairplot(sample_medium_high, hue="interest_level", markers='x', size = 1)
#sns.lmplot(x = "points", y= "price",  data=sample_medium_high, hue="interest_level")
sns.plt.show()

palette={'high': 'purple', 'medium': 'orange', 'low': 'red'}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.scatter(sample_low['score'], sample_low['price'], sample_low['score'], c ='r', marker='o')
ax.scatter(sample_medium['score'], sample_medium['price'], sample_medium['latitude'] + sample_medium['longitude'], c ='g', marker='^')
ax.scatter(sample_high['score'], sample_high['price'], sample_high['latitude'] + sample_high['longitude'], c ='b', marker='x')

ax.set_xlabel('score')
ax.set_ylabel('price')
ax.set_zlabel('latitude + longitude')

plt.show()

"""
