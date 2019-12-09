import os, numpy as np
from time import time
from gensim.models import KeyedVectors as Word2Vec
from sklearn.preprocessing import normalize
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer


def classes2embedding(dataset_name, class_name_inputs, wv_model):
    if dataset_name == 'ucf101':
        one_class2embed = one_class2embed_ucf
    elif dataset_name == 'hmdb51':
        one_class2embed = one_class2embed_hmdb
    elif dataset_name == 'kinetics':
        one_class2embed = one_class2embed_kinetics
    elif dataset_name == 'activitynet':
        one_class2embed = one_class2embed_activitynet
    elif dataset_name == 'sun':
        one_class2embed = one_class2embed_sun
    embedding = [one_class2embed(class_name, wv_model)[0] for class_name in class_name_inputs]
    embedding = np.stack(embedding)
    # np.savez('/workplace/data/motion_efs/home/biagib/ZeroShot/W2V_embedding/'+dataset_name,
    #          names=dataset_name, embedding=embedding)
    return normalize(embedding.squeeze())


def load_word2vec():
    try:
        wv_model = Word2Vec.load('/workplace/GoogleNews', mmap='r')
    except:
        wv_model = Word2Vec.load_word2vec_format(
            '/workplace/GoogleNews-vectors-negative300.bin', binary=True)
        wv_model.init_sims(replace=True)
        wv_model.save('assets/GoogleNews')
    return wv_model


def one_class2embed_ucf(name, wv_model):
    change = {
        'CleanAndJerk': ['weight', 'lift'],
        'Skijet': ['Skyjet'],
        'HandStandPushups': ['handstand', 'pushups'],
        'HandstandPushups': ['handstand', 'pushups'],
        'PushUps': ['pushups'],
        'PullUps': ['pullups'],
        'WalkingWithDog': ['walk', 'dog'],
        'ThrowDiscus': ['throw', 'disc'],
        'TaiChi': ['taichi'],
        'CuttingInKitchen': ['cut', 'kitchen'],
        'YoYo': ['yoyo'],
    }
    if name in change:
        name_vec = change[name]
    else:
        upper_idx = np.where([x.isupper() for x in name])[0].tolist()
        upper_idx += [len(name)]
        name_vec = []
        for i in range(len(upper_idx)-1):
            name_vec.append(name[upper_idx[i]: upper_idx[i+1]])
        name_vec = [n.lower() for n in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_hmdb(name, wv_model):
    change = {'claping': ['clapping']}
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.split(' ')
    name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_kinetics(name, wv_model):
    change = {
        'clean and jerk': ['weight', 'lift'],
        'dancing gangnam style': ['dance', 'korean'],
        'breading or breadcrumbing': ['bread', 'crumb'],
        'eating doughnuts': ['eat', 'bun'],
        'faceplanting': ['face', 'fall'],
        'hoverboarding': ['skateboard', 'electric'],
        'hurling (sport)': ['hurl', 'sport'],
        'jumpstyle dancing': ['jumping', 'dance'],
        'passing American football (in game)': ['pass', 'american', 'football', 'match'],
        'passing American football (not in game)': ['pass', 'american', 'football', 'park'],
        'petting animal (not cat)': ['pet', 'animal'],
        'punching person (boxing)': ['punch', 'person', 'boxing'],
        's head": 1}': ['head'],
        'shooting goal (soccer)': ['shoot', 'goal', 'soccer'],
        'skiing (not slalom or crosscountry)': ['ski'],
        'throwing axe': ['throwing', 'ax'],
        'tying knot (not on a tie)': ['ty', 'knot'],
        'using remote controller (not gaming)': ['remote', 'control'],
        'backflip (human)': ['backflip', 'human'],
        'blowdrying hair': ['dry', 'hair'],
        'making paper aeroplanes': ['make', 'paper', 'airplane'],
        'mixing colours': ['mix', 'colors'],
        'photobombing': ['take', 'picture'],
        'playing rubiks cube': ['play', 'cube'],
        'pretending to be a statue': ['pretend', 'statue'],
        'throwing ball (not baseball or American football)': ['throw',  'ball'],
        'curling (sport)': ['curling', 'sport'],
    }
    if name in change:
        name_vec = change[name]
    else:
        name = name.lower()
        name_vec_origin = name.split(' ')
        remove = ['a', 'the', 'of', ' ', '', 'and', 'at', 'on', 'in', 'an', 'or',
                  'do', 'using', 'with']
        name_vec = [n for n in name_vec_origin if n not in remove]

        not_id = [i for i, n in enumerate(name_vec) if n == '(not']
        if len(not_id) > 0:
            name_vec = name_vec[:not_id[0]]
        name_vec = [name.replace('(', '').replace(')', '') for name in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_activitynet(name_origin, wv_model):
    name_origin = name_origin if name_origin[0] != ' ' else name_origin[1:]
    change = {
        'Blow-drying_hair': ['dry', 'hair'],
        'Playing_rubik_cube': ['play', 'cube'],
        'Carving_jack-o-lanterns': ['carve', 'pumpkin'],
        'Mooping_floor': ['mop', 'floor'],
        'Ping-pong': ['table', 'tennis'],
        'Plataform_diving': ['diving', 'trampoline'],
        'Polishing_forniture': ['polish', 'furniture'],
        'Powerbocking': ['jump', 'shoes'],
        'Rock-paper-scissors': ['play', 'rock', 'paper', 'scissors'],
    }
    if name_origin in change:
        name_vec = change[name_origin]
    else:
        name = name_origin.lower()
        name = name.replace('_', ' ') # ACTIVITYNET
        name_vec_origin = name.split(' ')
        remove = ['a', 'the', 'of', ' ', '', 'and', 'at', 'on', 'in', 'an',
                  'do', 'using', 'with']
        name_vec = [n for n in name_vec_origin if n not in remove]
        not_id = [i for i, n in enumerate(name_vec) if n == '(not']
        if len(not_id) > 0:
            name_vec = name_vec[:not_id[0]]
        name_vec = [name.replace('(', '').replace(')', '') for name in name_vec]
        name_vec = verbs2basicform(name_vec)
    return wv_model[name_vec].mean(0), name_vec


def one_class2embed_sun(name, wv_model):
    change = {
        'car_interior/frontseat': ['car', 'interior', 'front', 'seat'],
        'forest/needleleaf': ['forest', 'needle', 'leaf'],
        'theater/indoor_procenium': ['theater', 'indoor'],
        'videostore': ['video', 'store'],
    }
    if name in change:
        name_vec = change[name]
    else:
        name_vec = name.lower().replace('/', '_').split('_')
    return wv_model[name_vec].mean(0), name_vec


def verbs2basicform(words):
    ret = []
    for w in words:
        analysis = wn.synsets(w)
        if any([a.pos() == 'v' for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'v')
        ret.append(w)
    return ret

