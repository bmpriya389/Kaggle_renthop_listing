import json
from process_features import *
import pandas
import numpy as np
from collections import Counter
from functools import reduce
from sklearn.metrics import log_loss

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
    features[i] = preprocess_features(data["features"][i])
    addresses[i] = ' '.join(data["display_address"][i].lower().split(' '))
    st_addresses[i] = ' '.join(data["street_address"][i].lower().split(' '))
    photos[i] = len(data["photos"][i])

features = split_feature(features)

samples_full = pandas.DataFrame({"bathrooms": data["bathrooms"], "bedrooms": data["bedrooms"],
                            "building_id": data["building_id"], "created": created,
                            "description": data["description"], "display_address": addresses,
                            "features": features, "latitude": data["latitude"],
                            "listing_id": data["listing_id"], "longitude": data["longitude"],
                            "manager_id": data["manager_id"], "photos": data["photos"], "price": data["price"],
                            "street_address": st_addresses, "interest_level": data["interest_level"],
                            "photos_num": photos})
columns_to_norm = ["bathrooms", "bedrooms", "created","longitude","latitude","price","photos_num"]
samples_full[columns_to_norm] = samples_full[columns_to_norm].apply(lambda x: (x-x.min())/(x.max()-x.min()))

samples = samples_full.head(round(49352*0.6))
t_samples = samples_full.loc[list(set(samples_full.index).difference(samples.index))]

"""
# Pre - process & construct testing dataset
t_created = dict().fromkeys(test["created"])
t_features = dict().fromkeys(test["features"])
t_addresses = dict().fromkeys(test["display_address"])
t_st_addresses = dict().fromkeys(test["street_address"])
t_photos = dict().fromkeys(test["photos"])

for i in t_created.keys():
    t_features[i] = preprocess_features(test["features"][i])
    t_addresses[i] = ' '.join(test["display_address"][i].lower().split(' '))
    t_st_addresses[i] = ' '.join(test["street_address"][i].lower().split(' '))
    t_photos[i] = len(test["photos"][i])

features = split_feature(features)

t_samples = pandas.DataFrame({"bathrooms": test["bathrooms"], "bedrooms": test["bedrooms"],
                              "building_id": test["building_id"], "created": t_created,
                              "description": test["description"], "display_address": t_addresses,
                              "features": t_features, "latitude": test["latitude"],
                              "listing_id": test["listing_id"], "longitude": test["longitude"],
                              "manager_id": test["manager_id"], "photos": test["photos"], "price": test["price"],
                              "street_address": t_st_addresses, "photos_num": t_photos})
"""

intrst_lvl = Counter(samples['interest_level'])

# processing raw features

# creating term frequency and interest level rate association
sample_high = samples.loc[(samples.interest_level == 'high')]
sample_medium = samples.loc[(samples.interest_level == 'medium')]
sample_low = samples.loc[(samples.interest_level == 'low')]

all_features_high = [x for l in sample_high["features"].tolist() for x in l]
all_features_medium = [x for l in sample_medium["features"].tolist() for x in l]
all_features_low = [x for l in sample_low["features"].tolist() for x in l]
all_features = Counter(all_features_high + all_features_medium + all_features_low)
all_features.update({'high': len(all_features_high), 'medium': len(all_features_medium),
                     'low': len(all_features_low)})

term_freq_high = term_freq_class(sample_high, len(sample_high), all_features.keys())
term_freq_medium = term_freq_class(sample_medium, len(sample_medium), all_features.keys())
term_freq_low = term_freq_class(sample_low, len(sample_low), all_features.keys())

attr_pt_class = attr_pts_high_med_low(term_freq_high, term_freq_medium, term_freq_low)

smoothing = 1

feature_liklihood = dict.fromkeys(all_features.keys(), 1)

for feat in feature_liklihood.keys():
    feature_liklihood[feat] = {int_lvl: (attr_pt_class[int_lvl][feat] + smoothing) /
                                        (all_features[int_lvl] + len(all_features))
                               for int_lvl in intrst_lvl.keys()}

feature_liklihood = pandas.DataFrame(feature_liklihood)


# apply this to all other posterior probabilities
feat_int_lvl_posterior = pandas.DataFrame({int_lvl: {i: reduce(lambda x, y: x * y,
                                                               [feature_liklihood[k][
                                                                    int_lvl] if k in feature_liklihood.columns else 1
                                                                for k in samples['features'][i]], 1)
                                                for i in samples.index} for int_lvl in intrst_lvl.keys()})
feat_int_lvl_posterior = (feat_int_lvl_posterior-feat_int_lvl_posterior.min())/(feat_int_lvl_posterior.max()-feat_int_lvl_posterior.min()) #normalizing
feat_int_lvl_posterior = feat_int_lvl_posterior.transpose()
# Processing street addresses
street_address = feat_liklihood('street_address', samples, intrst_lvl)
st_addr_post = street_address[0]

print('Street address done')

# processing addresses
address = feat_liklihood('display_address', samples, intrst_lvl)
addr_post = address[0]

print('address done')

# processing created
created = feat_liklihood('created', samples, intrst_lvl)
created_post = created[0]

print('created done')

# processing price
price = feat_liklihood('price', samples, intrst_lvl)
price_post = price[0]

print('price')

# processing building_id
building = feat_liklihood('building_id', samples, intrst_lvl)
build_post = building[0]

print('building')

# processing manager_id
manager = feat_liklihood('manager_id', samples, intrst_lvl)
manager_post = manager[0]

print('manager')

# Processing photos
photos_num = feat_liklihood('photos_num', samples, intrst_lvl)
photos_num_post = photos_num[0]

print('Photos num done')

# processing latitude
latitude = feat_liklihood('latitude', samples, intrst_lvl)
latitude_post = latitude[0]

print('latitude')

# processing longitude

longitude = feat_liklihood('longitude', samples, intrst_lvl)
longitude_post = longitude[0]

print('Longitude')


# processing bedrooms
bedrooms = feat_liklihood('bedrooms', samples, intrst_lvl)
bed_post = bedrooms[0]

print('Bedrooms')

# processing bathrooms
bathrooms = feat_liklihood('bathrooms', samples, intrst_lvl)
bath_post = bathrooms[0]

print('Bathrooms')

# processing combinations
# Considering max of feature probabilities
samples_results = pandas.concat([st_addr_post,
                                 addr_post,
                                 build_post,
                                 manager_post,
                                 price_post,
                                 created_post,
                                 # bed_bath_combo_post,
                                 # addr_combo_post,
                                 # latlon_combo_post,
                                 latitude_post,
                                 longitude_post,
                                 bed_post,
                                 # addressesprice_combo_post,
                                 bath_post,
                                 photos_num_post
                                 ], axis=1)

samples_results = samples_results.loc[:, ['street_addressmax',
                                          "building_idmax",
                                          "createdmax",
                                          #"addr_price_max",
                                          'display_addressmax',
                                          "manager_idmax"#,
                                          "latitudemax",
                                          "pricemax",
                                          "longitudemax",
                                          "bedroomsmax",
                                          "bathroomsmax",
                                          "photos_nummax"#,
                                          #"bedbath_max",
                                          #"latlon_max",
                                          #"sdaddr_max"
                                          ]]

samples_results = samples_results.max(axis=1)

print("evaluating!")
t_feat_int_lvl_posterior = pandas.DataFrame({i: {int_lvl: reduce(lambda x, y: x * y,
                                                [feature_liklihood[k][int_lvl] if k in feature_liklihood.columns else 1
                                                 for k in t_samples['features'][i]], 1)
                                for int_lvl in intrst_lvl.keys()} for i in t_samples.index})

t_feat_int_lvl_posterior = (t_feat_int_lvl_posterior - t_feat_int_lvl_posterior.min())/(t_feat_int_lvl_posterior.max()-t_feat_int_lvl_posterior.min()) #normalizing
t_feat_int_lvl_posterior = t_feat_int_lvl_posterior.transpose()
t_feat_int_lvl_posterior.replace(np.inf, 1).fillna( 1)
t_samples_saddr_result = (feat_posterior(t_samples, 'street_address', street_address[1], street_address[2],
                                         intrst_lvl) )
                                                                                                  

t_samples_addr_result = (feat_posterior(t_samples, 'display_address', address[1], address[2],
                                        intrst_lvl)  )


t_samples_created_result = (feat_posterior(t_samples, 'created', created[1], created[2],
                                        intrst_lvl)  )

t_samples_building_result = (feat_posterior(t_samples, 'building_id', building[1], building[2],
                                            intrst_lvl)  )
                                                                                                     

t_samples_mgr_result = (feat_posterior(t_samples, 'manager_id', manager[1], manager[2],
                                       intrst_lvl) )
                                                                                                

t_samples_price_result = (feat_posterior(t_samples, 'price', price[1], price[2], intrst_lvl)
                          )

t_samples_latitude_result = (feat_posterior(t_samples, 'latitude', latitude[1], latitude[2],
                                            intrst_lvl) )
                                                                                                     

t_samples_longitude_result = (feat_posterior(t_samples, 'longitude', longitude[1], longitude[2],
                                             intrst_lvl) )

t_samples_bedrooms_result = (feat_posterior(t_samples, 'bedrooms', bedrooms[1], bedrooms[2],
                                            intrst_lvl) )
                                                                                                     

t_samples_bathrooms_result = (feat_posterior(t_samples, 'bathrooms', bathrooms[1], bathrooms[2],
                                             intrst_lvl) )

t_samples_photos_no_result = (feat_posterior(t_samples, 'photos_num', photos_num[1], photos_num[2],
                                             intrst_lvl) )


t_samples_result1 = dict.fromkeys(intrst_lvl.keys(), {})
t_samples_result2 = dict.fromkeys(intrst_lvl.keys(), {})
t_samples_result3 = dict.fromkeys(intrst_lvl.keys(), {})

for int_lvl in intrst_lvl.keys():
    t_samples_result1[int_lvl] = {i: max([t_samples_saddr_result[int_lvl][i] * t_feat_int_lvl_posterior[int_lvl][i],
                                         # t_samples_addr_price_result[int_lvl][i],
                                         t_samples_addr_result[int_lvl][i]* t_feat_int_lvl_posterior[int_lvl][i],
                                         t_samples_created_result[int_lvl][i]* t_feat_int_lvl_posterior[int_lvl][i],
                                         #t_samples_addr_combo_result[int_lvl][i],
                                         t_samples_building_result[int_lvl][i]* t_feat_int_lvl_posterior[int_lvl][i]
                                         #t_samples_mgr_result[int_lvl][i],
                                         #t_samples_price_result[int_lvl][i],
                                         #t_samples_latitude_result[int_lvl][i],
                                         #t_samples_longitude_result[int_lvl][i],
                                         #t_samples_latlon_combo_result[int_lvl][i],
                                         #t_samples_bedrooms_result[int_lvl][i],
                                         #t_samples_bathrooms_result[int_lvl][i],
                                         #t_samples_bed_bath_combo_result[int_lvl][i],
                                         #t_samples_photos_no_result[int_lvl][i],
                                         ])
                                 for i in t_samples.index}

t_samples_result1 = pandas.DataFrame(t_samples_result1)
t_samples_result1["high"] = pandas.to_numeric(t_samples_result1["high"])
t_samples_result1["low"] = pandas.to_numeric(t_samples_result1["low"])
t_samples_result1["medium"] = pandas.to_numeric(t_samples_result1["medium"])

print("log loss: dcbm ",log_loss(y_true=t_samples["interest_level"], y_pred=t_samples_result1))

t_samples_result1['listing_id'] = pandas.Series(t_samples['listing_id'])
t_samples_result1 = t_samples_result1.reindex(columns=['listing_id', 'high', 'medium', 'low'])
