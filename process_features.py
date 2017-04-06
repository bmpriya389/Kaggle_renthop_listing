from collections import Counter
import jellyfish
from string import punctuation
import operator
from stemming.porter2 import stem
from functools import reduce
import re
import pandas


def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)


def jaccard_similarity(feature1, feature2):
    feat1 = set(feature1.split(" "))
    feat2 = set(feature2.split(" "))
    return len(feat1.intersection(feat2)) / len(feat1.union(feat2))


def rental_terms():
    amenities = ['yoga', 'gym', 'pilates', 'fitness', 'spa', 'dance', 'exercise', 'aerobic', 'cardio', 'elevator',
                 'doorman', 'valet', 'billiard', 'catering', 'facilities', 'golf', 'grill', 'grooming', 'theater',
                 'housekeeping', 'tv', 'surveillance', 'television', 'sofa', 'cable', 'satellite', 'concierge',
                 'volleyball', 'training', 'wifi', 'internet', 'tennis', 'parking', 'delivery', 'dry cleaning',
                 'intercom', 'medical', 'thermostat', 'entertainment room', 'garage', 'lifeguard', 'locker', 'jacuzzi',
                 'maid', 'ping pong', 'playroom', 'nursery', 'poker', 'ramp', 'sauna', 'service', 'wheelchair', 'bbq']
    utilities = ['laundry', 'washer', 'dryer', 'refrigerator', 'utilities', 'gas', 'electric', 'water',
                 'electricity', 'dishwasher', 'disposal', 'garbage', 'microwave', 'oven', 'heat',
                 'air condition', 'ac', 'appliances', 'cooler', 'manager']
    rental = ['no fee', 'superintendent', 'tenant', 'rental', 'underpriced', 'lease', 'realtor', 'concession',
              'discount', 'owner', 'free', 'deposit', 'incl', 'including', 'inunit', 'landlord', 'on site', 'livein',
              'move in']
    architecture = ['windows', 'french', 'venetian', 'pantry', 'soundproof', 'pool', 'swimming pool', 'ornate',
                    'parquet',
                    'penthouse', 'painted', 'fireplace', 'mosaic', 'custom', 'granite', 'hall', 'guest', 'studio',
                    'pre-war', 'post-war', 'built-in', 'office', 'foyer', 'oak', 'entrance', 'lobby', 'unique', 'rare',
                    'duplex', 'mansion', 'indoor', 'hardwood', 'fence', 'renovate', 'loft', 'maintained', 'european',
                    'murals', 'family', 'state-of-the-art', 'hookup', 'sunlight', 'historical', 'living', 'storage',
                    'barbecue', 'bbq', 'beamed', 'ceiling', 'dining', 'flooring', 'lobby', 'kitchen', 'sun', 'deck',
                    'recreational', 'linoleum', 'marble', 'misting', 'multilevel', 'molding', 'closet', 'lights',
                    'bright', 'limestone', 'counter', 'terrace', 'basement', 'court', 'glass', 'condo', 'unfurnished',
                    'yard', 'wardrobe', 'cabinets', 'triplex', 'redwood', 'backyard', 'balcony', 'recreation',
                    'immaculate', 'brick', 'townhouse', 'room', 'finish', 'queen', 'alcove', 'abundant',
                    'stainless steel',
                    'ss steel', 'ss', 'attached', 'outdoor']
    spatial = ['large', 'size', 'expansive', 'huge', 'enormous', 'massive', 'abundant', 'additional', 'kingsize',
               'ample', 'space', 'spacious', 'garden', 'patio', 'enclosed']
    age = ['new', 'brand new', 'modern']
    transport = ['train', 'bus', 'stop', 'transit', 'transport', 'subway', 'accessibility', 'citibikes', 'city']
    proximity = ['restaurants', 'on-site', 'viewing', 'near', 'park', 'blk', 'block', 'heart', 'market', 'hospital',
                 'shopping', 'library', 'atm', 'store', 'grocery', 'neighborhood', 'avenue', 'ave', 'beach', 'bicycle',
                 'bike', 'bridge', 'river', 'cafe']
    places = ['gramercy', 'greenpoint', 'brklyn', 'lexington', 'madison',
                 'manhattan', 'verizon', 'uptown', 'skyline', 'midtown', 'highrise', 'lowrise', 'murray', 'nyu',
                 'queens', 'soho']
    privacy = ['semi-private', 'private', 'share', 'personal', 'separate']
    pet = ['dog', 'cat', 'pet', 'animal', 'breeds']
    neg = ['no', 'non', 'not']

    return {'amenities': amenities, 'proximity': proximity, 'transport': transport, 'age': age, 'rental': rental,
            'architecture': architecture, 'utilities': utilities, 'privacy': privacy, 'pet': pet, 'order':
                ['amenities', 'transport', 'proximity', 'utilities', 'rental', 'architecture', 'age', 'privacy', 'pet'],
            'feature': utilities + transport + proximity + amenities + age + rental + architecture + spatial + privacy + pet}


def preprocess_features(feat):
    feature = [j.lower() for j in feat]
    http_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    phone_regex = '(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'
    email_regex = '[^@]+@[^@]+\.[^@]+'
    number_float = r"^\d*[.,]?\d*$"
    remove_non_ascii = r'[^\x00-\x7F]+'

    feature = [stem(strip_punctuation(
        re.sub(remove_non_ascii, '',
               re.sub('(\d+)', '',
                      re.sub(number_float, "random_number",
                             re.sub(email_regex, "email_addr",
                                    re.sub(phone_regex, "phone_no",
                                           re.sub(http_regex, "http_addr", word)))))))
    ) for word in feature]
    return feature


def split_feature(features):
    for j in features.keys():
        k = []
        for i in features[j]:
            k += re.split('\s{2,}|[*+•]', i)
        features[j] = [b.strip(" ") for b in k]

    return features


def terms(new_feat, all_feat):
    features_count = all_feat.most_common()
    features_count_keys = [i[0] for i in features_count] + new_feat

    features_found = dict.fromkeys(features_count_keys, {})
    similar_feat = dict.fromkeys(features_count_keys, {})

    for feature in features_count_keys:
        features_found[feature] = sorted(
            {comp: [max([max(map(lambda k: jellyfish.jaro_winkler(feature, k), comp.split(" "))),
                         max(map(lambda l: jellyfish.jaro_winkler(comp, l), feature.split(" ")))]),
                    jaccard_similarity(feature, comp)]
             for comp in features_count_keys}.items(),
            key=operator.itemgetter(1),
            reverse=True)

    """
    for k in sorted(list(features_found)):
        print("{:<30} {:<60}".format(k, str(features_found[k])))
    print(len(features_found))
    """

    for feat_grp in features_found:
        a = {}
        a[feat_grp] = {'score': 1.0, 'freq': all_feat[feat_grp], 'total': all_feat[feat_grp]}
        for x in range(len(features_found[feat_grp])):

            if features_found[feat_grp][x][1][0] < 0.86:
                break
            elif features_found[feat_grp][x][1][1] == 0:
                pass
            elif features_found[feat_grp][x][1][0] >= 0.86 and features_found[feat_grp][x][1][1] > 0:
                a[features_found[feat_grp][x][0]] = {'score': features_found[feat_grp][x][1][0],
                                                     'freq': all_feat[features_found[feat_grp][x][0]]}

        a['total'] = sum([a[i]['freq'] for i in a.keys()])
        similar_feat[feat_grp] = a

    all_rental_feat = {c: b for b in new_feat for c in similar_feat[b].keys()}

    for feat_grp in similar_feat.keys():
        if feat_grp in all_rental_feat.keys():
            similar_feat[feat_grp][all_rental_feat[feat_grp]] = 1

    """
    terms_common = rental_terms()['feature']

    for k in features_found.keys():
        features_found[k]['points'] = sum([_ for _ in terms_common if _ in k])

    for k in sorted(similar_feat.keys()):
        print("{:<30} {:<60} {}".format(k, str(similar_feat[k]['total']), similar_feat[k]))
    print(len(features_found))
    """

    new_feat_dictionary = {i: {'total': similar_feat[i]['total'], 'related': similar_feat[i]} for i in
                           similar_feat.keys()}

    """
    for k in sorted(new_feat):
        print("{:<30} {:<60}".format(k, str(new_feat_dictionary[k])))
    print(len(features_found))
    """

    if '' in new_feat_dictionary.keys():
        new_feat_dictionary['']['total'] = 0
    return new_feat_dictionary


def term_freq_class(samples, rate, key_terms):
    samples = samples.sample(frac=1)
    max_feat = dict.fromkeys(key_terms, 0)

    for i in range(0, len(samples), rate):
        sample = samples[i: i + rate]
        all_feat = reduce(lambda b, c: b + c, sample["features"], [])
        x = terms([], Counter(all_feat))
        max_feat = {feat: max([x[feat]['total'], max_feat[feat]]) if feat in x.keys() else max_feat[feat] for feat in
                    max_feat.keys()}
    max_feat = {feat: max_feat[feat] / rate for feat in max_feat.keys()}
    return max_feat


def attr_pts_high_med_low(term_freq_high, term_freq_medium, term_freq_low):
    feature_hml = pandas.DataFrame({'high': term_freq_high, 'medium': term_freq_medium, 'low': term_freq_low})
    feature_hml['max'] = feature_hml.max(axis=1)
    feature_hml['maxclass'] = feature_hml.idxmax(axis=1)
    return feature_hml
