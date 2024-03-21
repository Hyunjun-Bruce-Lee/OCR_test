# !sudo apt install tesseract-ocr
# !sudo apt-get install tesseract-ocr-kor
# !pip install pytesseract==0.3.9

import cv2
import re
import pandas as pd
import numpy as np
import pytesseract
from pytesseract import Output
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score

train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

def get_accuracy(answer_df, predict_df):
    return accuracy_score(answer_df['text'].values, predict_df['text'].values)

class PyTesseract:
    def __init__(self, lang='kor'):
        self.lang = lang
    
    def load_image(self, img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def text_preprocessing(self, text):
        text = text.replace('\n', '')
        text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]','', text)
        text = text.strip()
        return text
    
    def prediction(self, img_path_list):
        preds = []
        for img_path in tqdm(img_path_list):
            img = self.load_image(img_path)
            text = pytesseract.image_to_string(img, lang=self.lang)
            text = self.text_preprocessing(text)
            preds.append(text)
        print('Done.')
        return preds
    


tesseract_model = PyTesseract()

train_predicts = tesseract_model.prediction(train_df['img_path'].values)


train_predict_df = train_df.copy()
train_predict_df['text'] = train_predicts
print('Train Accuracy : ', get_accuracy(train_df, train_predict_df))


test_predicts = tesseract_model.prediction(test_df['img_path'].values)