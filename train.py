import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models.caption_model import build_caption_model
from utils.data_preprocessing import prepare_data
from utils.data_generator import CustomDataGenerator
from analysis_exploring import plot_learning_curve
import pickle

def train_model():
    image_path = 'flickr8k/Images'
    data_path = 'flickr8k/captions.txt'
    
    train, test, tokenizer, vocab_size, max_length, features, feature_extractor = prepare_data(data_path, image_path)
    
    train_generator = CustomDataGenerator(
        df=train, X_col='image', y_col='caption', batch_size=64, directory=image_path,
        tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_length, features=features
    )
    validation_generator = CustomDataGenerator(
        df=test, X_col='image', y_col='caption', batch_size=64, directory=image_path,
        tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_length, features=features
    )
    
    caption_model = build_caption_model(vocab_size, max_length)
    
    checkpoint = ModelCheckpoint(
        "model.keras",
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=0.00000001)
    
    history = caption_model.fit(
        train_generator, epochs=50, validation_data=validation_generator,
        callbacks=[checkpoint, earlystopping, learning_rate_reduction]
    )
    
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    feature_extractor.save("feature_extractor.keras")
    
    plot_learning_curve(history)
    
    return history

if __name__ == "__main__":
    history = train_model()