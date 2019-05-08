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
parser.add_argument("instrument_num")
args = parser.parse_args()
INPUTPATH = args.inputpath
INSTRU_NUM = int(args.instrument_num)

def keys_of_value(dct, value):
    for k in dct:
        if isinstance(dct[k], list):
            if value in dct[k]:
                return k
        else:
            if value == dct[k]:
                return k

instru_dict = {"voi":10, "vio":9, "tru":8, "sax":7, "pia":6, "org":5, "gel":4, "gac":3, "flu":2, "cla":1, "cel":0}

bst = xgb.Booster({'nthread':4}) #init model
bst.load_model("xgboost.model.txt")

txt_path = INPUTPATH[:-3]+'txt'
music_class = open(txt_path)
instrument_type_true = list(filter(None, music_class.read().split('\t\n')))

#输入多乐器乐曲，得到其乐器的得分
from collections import Counter
def multiMusicExactPrediction(file_path, bst, instrument_num):
    y, sr = librosa.load(file_path)
    m_slaney = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, dct_type=2)
    single_df = pd.DataFrame(m_slaney.T)
    result = bst.predict(xgb.DMatrix(single_df))
    return_pred = []
    predictions = []
    for x in result:
        predictions.append(np.argmax(x))
    counter_words = Counter(predictions)
    most_counter = counter_words.most_common(instrument_num)
    for j in most_counter:
        return_pred.append((keys_of_value(instru_dict, j[0])))
    return(return_pred)

print("the original music is: ", instrument_type_true)
print("our prediction is: ", multiMusicExactPrediction(INPUTPATH, bst, INSTRU_NUM))
