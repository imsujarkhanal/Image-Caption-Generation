import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import wrap
import pandas as pd
from utils.data_preprocessing import readImage


def display_images(temp_df, image_path):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(15, 15))
    n = 0
    for i in range(15):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.9, wspace=0.5)
        image = readImage(f"{image_path}/{temp_df.image[i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")
    plt.show()

def plot_learning_curve(history):
    plt.figure(figsize=(20, 8))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

def explore_data(data_path, image_path):
    data = pd.read_csv(data_path)
    display_images(data.sample(15), image_path)

if __name__ == "__main__":
    data_path = 'flickr8k/captions.txt'
    image_path = 'flickr8k/images'
    explore_data(data_path, image_path)