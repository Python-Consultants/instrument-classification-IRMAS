import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import xgboost as xgb
import argparse


from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("inputpath")
args = parser.parse_args()
INPUTPATH = args.inputpath

def keys_of_value(dct, value):
    for k in dct:
        if isinstance(dct[k], list):
            if value in dct[k]:
                return k
        else:
            if value == dct[k]:
                return k

instru_dict = {"voi":10, "vio":9, "tru":8, "sax":7, "pia":6, "org":5, "gel":4, "gac":3, "flu":2, "cla":1, "cel":0}

#输入单乐器乐曲，得到其乐器的得分
from collections import Counter
def musicFile2FeatureTable(file_path):
    txt_path = file_path[:-3]+'txt'
    music_class = open(txt_path)
    instrument_type_true = music_class.read()
    print('the instrument is %s'%instrument_type_true)
    y, sr = librosa.load(file_path)
    m_slaney = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, dct_type=2)
    single_df = pd.DataFrame(m_slaney.T)
    bst = xgb.Booster({'nthread':4}) #init model
    bst.load_model("xgboost.model.txt")
    result = bst.predict(xgb.DMatrix(single_df))
    predictions = []
    for x in result:
        predictions.append(np.argmax(x))
    counter_words = Counter(predictions)
    most_counter = counter_words.most_common(1)
    return(keys_of_value(instru_dict, most_counter[0][0]))

print("our prediction is %s"%musicFile2FeatureTable(INPUTPATH))
