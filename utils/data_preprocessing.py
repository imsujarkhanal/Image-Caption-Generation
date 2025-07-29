import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tqdm import tqdm
import os
from utils.file_utils import validate_image_path

def read_image(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img / 255.0
    return img

def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+", " "))
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(str(word)) > 1]))
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

def extract_features(images, image_path, feature_extractor, img_size=224):
    features = {}
    for image in tqdm(images, desc="Extracting image features"):
        full_path = validate_image_path(image, image_path)
        img = read_image(full_path)
        img = np.expand_dims(img, axis=0)
        feature = feature_extractor.predict(img, verbose=0)
        features[image] = feature
    return features

def prepare_data(data_path, image_path, feature_extractor):
    data = pd.read_csv(data_path)
    data = text_preprocessing(data)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(data['caption'].tolist())
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in data['caption'].tolist())
    
    images = data['image'].unique().tolist()
    nimages = len(images)
    split_index = round(0.85 * nimages)
    train_images = images[:split_index]
    val_images = images[split_index:]
    
    train = data[data['image'].isin(train_images)].reset_index(drop=True)
    test = data[data['image'].isin(val_images)].reset_index(drop=True)
    
    features = extract_features(images, image_path, feature_extractor)
    
    return train, test, tokenizer, vocab_size, max_length, features