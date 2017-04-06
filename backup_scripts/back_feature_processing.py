from collections import Counter
import jellyfish
import numpy

def jaccard_similarity(feature1, feature2):
    feat1 = set(feature1)
    feat2 = set(feature2)
    return len(feat1.intersection(feat2))/len(feat1.union(feat2))

def rental_terms():
    amenties_pts = ['yoga', 'gym', 'elevator', 'doorman', 'valet']
    proximity_pts = ['blks', 'blocks', 'blk', 'steps']
    transport_pts = ['train', 'bus', 'bus stop', 'stop']
    age_pts = ['new', 'brand new', 'modern']
    architecture_pts = ['windows', 'studio', 'pre-war', 'spacious', 'built-in', 'large', 'expansive', 'duplex', 'mansion', 'indoor']
    utilities_pts = ['laundry', 'washer', 'dryer', 'utilities', 'bbq']
    rental_pts = ['no fee']
    pet_pts = ['dogs', 'cats', 'pets']
    neg_pts = ['no', 'non', 'not']

def feature_process(feat_dict):

    split_merged_features = []
    delete_keys = []
    for i in feat_dict.keys():
        if "*" in i:
            delete_keys.append(i)
            split_merged_features.extend(i.split("*"))
        if "+" in i:
            delete_keys.append(i)
            split_merged_features.extend(i.split("+"))
        if u"•" in i:
            delete_keys.append(i)
            split_merged_features.extend(i.split("•"))

    feat_from_merged = Counter(split_merged_features)
    feat_dict.update(feat_from_merged)
    feat_dict.pop('', None)

    for i in delete_keys:
        feat_dict.pop(i, None)

    feat_distance = dict.fromkeys(feat_dict)

    for i in feat_dict.keys():
        feat_distance[i] = [j for j in feat_dict.keys() if jellyfish.jaro_distance(i, j)>=0.8 or jaccard_similarity(i, j) >= 0.80]
        feat_distance[i].remove(i)

    existing_keys=[]
    delete_keys=[]

    for i in feat_distance.keys():
        if i in existing_keys:
            delete_keys.append(i)
        else:
            existing_keys.extend(feat_distance[i])

    for i in delete_keys:
        feat_distance.pop(i, None)


    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(feat_distance.keys())):
        print("{:<30} {:<60}".format(k, '   '.join(feat_distance[k]['total'])))

    print(len(feat_distance))
    return feat_dict



from collections import Counter
import jellyfish
import operator

def jaccard_similarity(feature1, feature2):
    feat1 = set(feature1)
    feat2 = set(feature2)
    return len(feat1.intersection(feat2))/len(feat1.union(feat2))

def shared_terms(feature1, feature2):
    return set(feature1.split(" ")).intersection(set(feature2.split(" ")))

def rental_terms():

    #count_1grams = list_term_freq(feat_dict)

    amenities = ['yoga', 'gym', 'pilates', 'fitness', 'elevator', 'doorman', 'valet', 'billiard',
                 'housekeeping', 'tv', 'television', 'concierge', 'training', 'wifi']
    utilities = ['laundry', 'washer', 'dryer', 'utilities', 'bbq', 'gas', 'electric', 'water', 'dishwasher',
                 'microwave', 'oven', 'heater']
    rental = ['no fee', 'underpriced', 'concession', 'discount', 'free']
    architecture = ['windows', 'french', 'fireplace', 'studio', 'pre-war', 'spacious', 'built-in', 'large', 'expansive',
                    'massive', 'unique', 'rare', 'duplex', 'mansion', 'indoor', 'hardwood', 'fence', 'renovate',
                    'european', 'state-of-the-art', 'sunlight', 'historical', 'living-room', 'storage', 'lobby',
                    'kitchen', 'sundecks', 'recreational', 'closet', 'lights', 'bright', 'counter', 'terrace',
                    'condo', 'yard', 'brick', 'room', 'finish']
    age = ['new', 'brand new', 'modern']
    transport = ['train', 'bus', 'stop', 'transport']
    proximity = ['restaurants', 'on-site', 'near', 'park']
    privacy = ['semi-private', 'private', 'share', 'personal']
    pet = ['dogs', 'cats', 'pets']
    neg = ['no', 'non', 'not']

    return {'amenities': amenities, 'proximity': proximity, 'transport': transport, 'age': age, 'rental' :rental,
            'architecture': architecture, 'utilities': utilities, 'privacy': privacy, 'pet': pet, 'order':
            ['amenities', 'utilities', 'rental', 'architecture', 'age', 'transport', 'proximity', 'privacy', 'pet'],
            'feature': amenities + proximity + transport + age + rental + architecture + utilities + privacy + pet}

def list_term_freq(feat_dict):
    split_merged_features = []
    for i in feat_dict.keys():
        split_merged_features.extend(i.split(" "))
    return Counter(split_merged_features)

def feature_process(feat_dict):

    split_merged_features = []
    delete_keys = []
    for i in feat_dict.keys():
        if "*" in i:
            delete_keys.append(i)
            split_merged_features.extend(i.split("*"))
        if "+" in i:
            delete_keys.append(i)
            split_merged_features.extend(i.split("+"))
        if u"•" in i:
            delete_keys.append(i)
            split_merged_features.extend(i.split("•"))


    features_fixed = rental_terms()
    feat_from_merged = Counter(split_merged_features)
    feat_from_merged.pop('')
    feat_from_merged.pop('')


    feat_distance = dict.fromkeys(features_fixed['feature'], [])
    feat_term_similarity = dict.fromkeys(features_fixed['feature'], [])

    for i in features_fixed['order']:
        for l in features_fixed[i]:
            feat_term_similarity[l] = sorted({j: max([jellyfish.jaro_winkler(l,k) for k in j.split(" ")]) for j in feat_from_merged.keys()}.items(), key=lambda x: (-x[1]))


    for i in features_fixed['order']:
        for l in features_fixed[i]:
            feat_distance[l] = [j for j in feat_from_merged.keys() if len(shared_terms(l, j)) >= 1 or (jaccard_similarity(l,j)>0.7 and l in j)]
            for k in feat_distance[l]:
                feat_from_merged.pop(k, None)
            #feat_distance[l].extend([j for j in feat_from_merged.keys() if max([jellyfish.jaro_winkler(l,k) for k in j.split(" ")]) >= 0.915])
            feat_distance[l].extend([j for j in feat_from_merged.keys() if max([jellyfish.jaro_winkler(l, k.replace("/","")) for k in j.split(" ")]) >= 0.9142])
                                     #max([jellyfish.jaro_winkler(l, k.replace("/","")) for k in j.split(" ")]) >= 0.902])
            for k in feat_distance[l]:
                feat_from_merged.pop(k, None)
     #remove / & ed tions



    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(feat_term_similarity.keys())):
        print("{:<30} {:<60}".format(k, str(feat_term_similarity[k])))


    print("")
    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(feat_distance.keys())):
        print("{:<30} {:<60}".format(k, str(feat_distance[k])))


    print(len(feat_from_merged.keys()))
    print(feat_from_merged.keys())
    print(jellyfish.jaro_winkler('renovate', 'renovations'))
    return list_term_freq(feat_dict)




    features_count = feat_dict.most_common()
    features_count_keys = [i[0] for i in features_count]
    features_found = dict.fromkeys(feat_dict.keys(),{'related': [], 'total': 0})
    merged = []
    print(len(features_count))
    for i in features_count:
        for j in features_count_keys:
            if j == i[0]:
                features_found[i[0]] = {'related': [i[0]], 'total': i[1]}
            else:
                for k in i[0].split(" "):
                    if jellyfish.jaro_distance(strip_punctuation(k),j) >= 0.9 and i[0] not in set(features_found[j]['related']):
                        features_found[j]['related'].append(i[0])
                        merged.append(i[0])
                        features_found[j]['total'] += i[1]
                        break;

    deleted = set([i for i in features_found.keys() if features_found[i]['total'] == 1]).intersection(merged)

    for i in deleted:
        features_found.pop(i, None)
    features_found['']['related'] = []
    features_found['']['total'] = 0

    expand_grid = {}
    for i in features_found.keys():
        expand_grid.update(dict.fromkeys(features_found[i]['related'], {'related': features_found[i]['related'], 'total': features_found[i]['total']}))

    features_found.update(expand_grid)


    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(features_found.keys())):
        print("{:<30} {:<60}".format(k, str(features_found[k])))

    merge_factor2 = []
    for i in features_found:
        merge_factor2.extend(i.split(" "))

    merge_factor2 = Counter(merge_factor2)
    merge_factor2_count = merge_factor2.most_common()
    merge_factor2_count_keys = [i[0] for i in merge_factor2_count]
    merged_features_found = dict.fromkeys(merge_factor2.keys(), {'related': [], 'total': 0})
    merged = []

    print(merge_factor2)
    print(merge_factor2_count)
    print(merge_factor2_count_keys)

    print(len(merge_factor2_count))

    for i in merge_factor2_count:
        for j in merge_factor2_count_keys:
            if j == i[0]:
                merged_features_found[i[0]] = {'related': [i[0]], 'total': i[1]}
            elif jellyfish.jaro_distance(strip_punctuation(i[0]), j) >= 0.9 and i[0] not in set(merged_features_found[j]['related']):
                merged_features_found[j]['related'].append(i[0])
                merged.append(i[0])
                merged_features_found[j]['total'] += i[1]

                break
            else:
                pass

    deleted = set([i for i in merged_features_found.keys() if merged_features_found[i]['total'] == 1]).intersection(merged)
    for i in deleted:
        merged_features_found.pop(i)

    print("{:<30} {:<60}".format('feature', 'similar keys'))
    for k in sorted(list(merged_features_found.keys())):
        print("{:<30} {:<60}".format(k, str(merged_features_found[k])))
    print(len(merged_features_found))
    print(punctuation)
