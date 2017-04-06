import json
import datetime
import pandas
from collections import Counter
from process_features import *
from matplotlib import pyplot as plt
import matplotlib
from functools import reduce
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math

sns.set()

matplotlib.style.use('ggplot')

data = json.loads(open("Data/train.json").read())
test = json.loads(open("Data/test.json").read())

created = dict().fromkeys(data["created"])
features = dict().fromkeys(data["features"])
interest_level = dict().fromkeys(data["interest_level"])

for i in created.keys():
    created[i] = (datetime.datetime.now() - datetime.datetime.strptime(data["created"][i], "%Y-%m-%d %H:%M:%S")).days
    features[i] = preprocess_features(data["features"][i]) + preprocess_features(data["description"][i])

features = split_feature(features)

samples = pandas.DataFrame({"bathrooms": data["bathrooms"], "bedrooms": data["bedrooms"],
                            "building_id": data["building_id"], "created": created,
                            "description": data["description"], "display_address": data["display_address"],
                            "features": features, "latitude": data["latitude"],
                            "listing_id": data["listing_id"], "longitude": data["longitude"],
                            "manager_id": data["manager_id"], "photos": data["photos"], "price": data["price"],
                            "street_address": data["street_address"], "interest_level": data["interest_level"]})

samples = samples.loc[(samples["price"] <= 80000)]

# normalize
cols_to_norm = ['bathrooms', 'bedrooms', 'latitude', 'longitude', 'price']
samples[cols_to_norm] = samples[cols_to_norm].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

sample_high = samples.loc[(samples.interest_level == 'high')]
sample_medium = samples.loc[(samples.interest_level == 'medium')]
sample_low = samples.loc[(samples.interest_level == 'low')]

all_features = Counter([x for l in samples["features"].tolist() for x in l])

rental_term = rental_terms()
proximity = rental_term['proximity']
utilities = rental_term['utilities']
amenities = rental_term['amenities']

new_feat = proximity + utilities + amenities

term_freq = terms(new_feat, all_features)
new_feat_freq = {i: term_freq[i]['related'] for i in new_feat}

term_freq_high = term_freq_class(sample_high, 1000, all_features.keys())
term_freq_medium = term_freq_class(sample_medium, 1000, all_features.keys())
term_freq_low = term_freq_class(sample_low, 1000, all_features.keys())

high_features = Counter([x for l in sample_high["features"].tolist() for x in l])
medium_features = Counter([x for l in sample_medium["features"].tolist() for x in l])
low_features = Counter([x for l in sample_low["features"].tolist() for x in l])

common_hm = set(high_features.keys()).intersection(medium_features.keys())
not_comm_hm = set(all_features.keys()).difference(common_hm)
common_hml = common_hm.intersection(low_features.keys())

attr_pt_class = attr_pts_high_med_low(term_freq_high, term_freq_medium, term_freq_low)

smoothing = 1

points = dict.fromkeys(samples.index, 1)
score = dict.fromkeys(samples.index, 1)
class_weight = {'high': 100, 'medium': 10, 'low': 1}

feat_add = pandas.DataFrame(dict.fromkeys(new_feat, dict.fromkeys(samples.index, 0)))

for i in samples.index:
    points[i] = reduce(lambda x, y: x + y, [all_features[k] for k in samples["features"][i]], 1)\
                / len(all_features)
    score[i] = reduce(lambda a, b: a + b, [(all_features[k] * attr_pt_class['max'][k] +
                                            class_weight[attr_pt_class['maxclass'][k]])
                                           if k not in common_hm
                                           else (all_features[k] * attr_pt_class['max'][k])
                                           for k in samples['features'][i]], 1)
    for feat in new_feat:
        for k in samples['features'][i]:
            if k in new_feat_freq[feat].keys():
                feat_add[feat][i] = 1 + attr_pt_class['max'][k]
                break

samples["points"] = pandas.Series(points)
samples["score"] = pandas.Series(score)
samples["proximity"] = feat_add[proximity].sum(axis=1)
samples["utilities"] = feat_add[utilities].sum(axis=1)

samples_stat = samples.groupby('interest_level')

print(samples_stat["points"].describe())
print(samples_stat["score"].describe())
print(samples_stat["price"].describe())

print(Counter(samples["interest_level"]))

sample_high = samples.loc[(samples.interest_level == 'high')]
sample_medium = samples.loc[(samples.interest_level == 'medium')]
sample_low = samples.loc[(samples.interest_level == 'low')]

sample_low_medium =([sample_low, sample_medium])
sample_medium_high = pandas.concat([sample_high, sample_medium])

new_feat.append('interest_level')
"""
#sns.pairplot(sample_medium_high, markers='x', hue='interest_level', palette={'high': 'purple', 'medium': 'orange', 'low': 'red'}, size = 1)
sns.pairplot(samples[new_feat], hue="interest_level", markers='x', size = 0.5)
#sns.lmplot(x = "points", y= "price",  data=sample_medium_high, hue="interest_level")
sns.plt.show()

palette = {'high': 'purple', 'medium': 'orange', 'low': 'red'}
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(sample_high['proximity'] ,sample_high['utilities'], sample_high['score'], sample_high['points'], c='b', marker='x')
ax.scatter(sample_medium['proximity'] , sample_medium['utilities'], sample_medium['score'], sample_medium['points'],c='g', marker='x')

#ax.scatter(sample_low['points'], sample_low['latitude'] * sample_low['longitude'], sample_low['price'], c ='r', marker='o')
#ax.scatter(sample_medium['points'], sample_medium['latitude'], sample_medium['price'], c ='g', marker='^')
#ax.scatter(sample_high['points'], sample_high['latitude'], sample_high['price'], c ='b', marker='x')

ax.set_xlabel('proximity & utilities')
ax.set_ylabel('score')
ax.set_zlabel('points')


plt.show()

"""
"""