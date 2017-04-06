from collections import Counter
import jellyfish
from string import punctuation
import operator
from stemming.porter2 import stem

def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def jaccard_similarity(feature1, feature2):
    feat1 = set(feature1.split(" "))
    feat2 = set(feature2.split(" "))
    return len(feat1.intersection(feat2))/len(feat1.union(feat2))

def rental_terms():
    amenities = ['yoga', 'gym', 'pilates', 'fitness', 'spa', 'dance','exercise', 'aerobic', 'cardio', 'elevator',
                 'doorman', 'valet', 'billiard', 'catering', 'facilities', 'golf', 'grill', 'grooming', 'theater',
                 'housekeeping', 'tv','surveillance', 'television', 'sofa','cable', 'satellite','concierge',
                 'volleyball', 'training', 'wifi', 'internet', 'tennis', 'parking', 'delivery', 'dry cleaning',
                 'intercom', 'medical', 'thermostat', 'entertainment room', 'garage' ,'lifeguard', 'locker',
                 'maid', 'ping pong', 'playroom', 'nursery', 'poker', 'ramp', 'sauna', 'service', 'wheelchair']
    utilities = ['laundry', 'washer', 'dryer', 'refrigerater','utilities', 'bbq', 'gas', 'electric', 'water',
                 'dishwasher', 'disposal', 'garbage', 'jacuzzi', 'keycard', 'key', 'microwave', 'oven', 'heat',
                 'air conditioning','appliances', 'a/c', 'cooler', 'manager','movein']
    rental = ['no fee', 'superintendent','tenant', 'rental','underpriced', 'lease','realtor','concession', 'discount',
              'owner','free', 'deposit', 'incl', 'including', 'inunit', 'landlord', 'on site', 'livein']
    architecture = ['windows', 'french','venetian', 'pantry','soundproof','pool', 'swimming pool','ornate','parquet',
                    'penthouse', 'painted', 'fireplace','mosaic', 'custom', 'granite','hall','guest', 'studio',
                    'pre-war', 'post-war', 'built-in', 'office','foyer','oak', 'entrance', 'lobby', 'unique', 'rare',
                    'duplex', 'mansion', 'indoor', 'hardwood', 'fence', 'renovate','loft', 'maintained', 'european',
                    'murals','family','state-of-the-art', 'hookup', 'sunlight', 'historical', 'living', 'storage',
                    'barbecue', 'bbq','beamed', 'ceiling', 'dining', 'flooring', 'lobby', 'kitchen', 'sun','deck',
                    'recreational',  'linoleum','marble', 'misting', 'multilevel', 'molding', 'closet', 'lights',
                    'bright', 'limestone', 'counter', 'terrace', 'basement', 'court', 'glass', 'condo','unfurnished',
                    'yard', 'wardrobe', 'cabinets', 'triplex', 'redwood', 'backyard', 'balcony', 'recreation',
                    'immaculate','brick','townhouse', 'room', 'finish','queen', 'alcove', 'abundant', 'stainless steel',
                    'ss steel', 'ss','attached', 'outdoor']
    spatial = ['large', 'size', 'expansive', 'huge', 'enormous', 'massive', 'abundant', 'additional', 'kingsize',
               'ample', 'space', 'spacious', 'garden', 'patio', 'enclosed']
    age = ['new', 'brand new', 'modern']
    transport = ['train', 'bus', 'stop', 'transit','transport', 'subway', 'accessibility', 'extravagant', 'spacious',
                 'citibikes', 'city']
    proximity = ['restaurants', 'on-site', 'viewing', 'near', 'park', 'blk','block', 'heart', 'market', 'hospital',
                 'shopping', 'library', 'atm', 'store', 'grocery', 'neighborhood','avenue', 'ave','beach','bicycle',
                 'bike', 'bridge', 'river', 'cafe', 'gramercy', 'greenpoint','brklyn', 'lexington', 'madison',
                 'manhattan','verizon','uptown', 'skyline', 'midtown', 'highrise', 'lowrise', 'murray', 'nyu','queens',
                 'soho']
    privacy = ['semi-private', 'private', 'share', 'personal', 'separate']
    pet = ['dog', 'cat', 'pet', 'animal', 'breeds']
    neg = ['no', 'non', 'not']

    return {'amenities': amenities, 'proximity': proximity, 'transport': transport, 'age': age, 'rental' :rental,
            'architecture': architecture, 'utilities': utilities, 'privacy': privacy, 'pet': pet, 'order':
            ['amenities', 'transport', 'proximity', 'utilities', 'rental', 'architecture', 'age',  'privacy', 'pet'],
            'feature': utilities + transport + proximity + amenities+ age + rental + architecture + spatial + privacy + pet}


def split_feature(features):
    for j in features.keys():
        k = []
        for i in features[j]:
            if "*" in i:
                k.extend(i.split("*"))
            if "+" in i:
                k.extend(i.split("+"))
            if "•" in i:
                k.extend(i.split("•"))
            if "*" not in i and "+" not in i and "•" not in i:
                k.append(i)
            if '' in k:
                k.remove('')

        features[j] = [b.strip(" ") for b in k]

    return features

def terms(feat_dict):
    features_count = feat_dict.most_common()
    features_count_keys = [i[0] for i in features_count]

    features_found = dict.fromkeys(features_count_keys, {'related':[]})

    for feature in features_count_keys:
        features_found[feature] = sorted({comp: max([max(map(lambda k: jellyfish.jaro_winkler(feature, k),
                                                             comp.split(" "))), jaccard_similarity(feature, comp)])
                                          for comp in features_count_keys}. items(), key = operator.itemgetter(1),
                                         reverse = True)

    for feat_grp in features_found:
        a = {}
        a[feat_grp] = {'score': 1.0, 'freq': feat_dict[feat_grp]}
        for x in range(1, len(features_found[feat_grp])):
            if stem(features_found[feat_grp][x][0].split(" ")[0]) in feat_grp or stem(feat_grp.split(" ")[0]) in features_found[feat_grp][x][0]:
                a[features_found[feat_grp][x][0]] = {'score': features_found[feat_grp][x][1], 'freq': feat_dict[features_found[feat_grp][x][0]]}
                #a[features_found[feat_grp][x][0]] = {'freq': feat_dict[features_found[feat_grp][x][0]]}
            elif features_found[feat_grp][x][1] <= 0.80:
                break
            elif features_found[feat_grp][0][1] - features_found[feat_grp][x][1] <= 0.08 :
                a[features_found[feat_grp][x][0]] = {'score': features_found[feat_grp][x][1], 'freq': feat_dict[features_found[feat_grp][x][0]]}
                #a[features_found[feat_grp][x][0]] = {'freq': feat_dict[features_found[feat_grp][x][0]]}
        a['total'] = sum([a[i]['freq'] for i in a.keys()])
        features_found[feat_grp] = a

    print("--------------------------------------------------------------")

    """
    for k in sorted(list(features_found.keys())):
        print("{:<30} {:<60}".format(k, str(features_found[k])))
    print(len(features_found))
    """
    terms_common = rental_terms()['feature']

    for k in features_found.keys():
        features_found[k]['points'] = sum([100 for _ in terms_common if _ in k])

    """
    for k in sorted(list(term_dict.keys())):
        print("{:<30} {:<60}".format(k, str(term_dict[k])))
    print(len(term_dict))
    """

    new_feat_dictionary = {i: {'total': features_found[i]['total'], 'points': features_found[i]['points']} for i in features_found.keys()}

    return new_feat_dictionary


def feature_process(feat_dict, all_features):
    features_fixed = rental_terms()
    feat_from_merged = Counter(all_features)
    feat_from_merged.pop('')
    remove_keys = []
    for i in feat_from_merged.keys():
        if 'bath' in i or 'br' in i or 'bed' in i:
            remove_keys.append(i)

    for i in remove_keys:
        feat_from_merged.pop(i, None)

    feat_distance = dict.fromkeys(features_fixed['feature'], [])
    feat_term_similarity = dict.fromkeys(features_fixed['feature'], [])
    feat_term_similarity_max = dict.fromkeys(features_fixed['feature'], [])

    for l in features_fixed['feature']:
        feat_term_similarity[l]= {j: max([jellyfish.jaro_winkler(l, k.replace("/", "")) for k in j.split(" ")]) for j in feat_from_merged.keys()}
        feat_term_similarity_max[l] = sorted(feat_term_similarity[l].items(), key = lambda x: -x[1])

    for j in reversed(range(86, 102, 2)):
        for l in features_fixed['feature']:
            if feat_term_similarity_max[l][0][1]>= j/100 or l in feat_term_similarity_max[l][0][0]:
                feat_distance[l] = feat_distance[l] + [feat_term_similarity_max[l][0][0]]
                for k in features_fixed['feature']:
                    feat_term_similarity[k].pop(feat_term_similarity_max[l][0][0], None)
                    feat_from_merged.pop(feat_term_similarity_max[l][0][0], None)
                    feat_term_similarity_max[k] = sorted(feat_term_similarity[k].items(), key=lambda x: -x[1])


    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(feat_term_similarity.keys())):
        print("{:<30} {:<60}".format(k, str(feat_term_similarity[k])))

    print("")
    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(feat_distance.keys())):
        print("{:<30} {:<60}".format(k, str(feat_distance[k])))
    print(len(feat_from_merged.keys()))
    print(feat_from_merged.keys())


    return feat_dict

