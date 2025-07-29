# Image Captioning Project

This project implements an image captioning model using a CNN (DenseNet201) for feature extraction and an LSTM for caption generation, with a Streamlit app for user interaction.

![dog](https://github.com/user-attachments/assets/462a6cd4-ac88-47ec-9d4f-6cac47477f7c)

## Setup

1. **Clone the repository**

   ```bash
   git clone <repository_url>
   cd image_captioning_project
   ````

2. **Create and activate a virtual environment** (optional but recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Flickr8k dataset** and place it in the `data/flickr8k/` directory.

## Data Exploration

Before training, explore the dataset to visualize sample images and captions:

```bash
python analysis_exploring.py
```

## Training

Train the model and plot the learning curve:

```bash
python train.py
```

## Running the Streamlit App

Launch the Streamlit app to generate captions in real-time:

```bash
streamlit run app.py
```

## Project Structure

- **models**: Contains model definitions for feature extraction and captioning
- **utils**: Utility functions for data preprocessing,caption generator, data generation, and file utilities
- **data**: Directory for the Flickr8k dataset (not included in the repository)
- **app.py**: Streamlit application for generating captions
- **train.py**: Script to train the model and plot the learning curve
- **analysis_exploring.py**: Script to explore the dataset and visualize data
- **requirements.txt**: Project dependencies

## Model Choice Rationale

* **DenseNet201 for Feature Extraction**: DenseNet201 captures hierarchical feature representations through dense blocks, producing robust 1920-dimensional image embeddings.
* **LSTM for Caption Generation**: LSTM networks effectively model sequential dependencies, enabling coherent word-by-word caption generation based on image embeddings and prior context.

## Notes

* Ensure dataset paths in `train.py`, `app.py`, and `analysis_exploring.py` match your local directory structure.
* Link of [Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) I used
* You can extend this to larger datasets like Flickr30k or MS-COCO for improved performance.
* Consider using GPU acceleration for faster training.
