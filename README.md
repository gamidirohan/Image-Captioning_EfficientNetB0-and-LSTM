# Image Captioning with EfficientNet and LSTM

## Project Overview

This project implements an image captioning model that uses a combination of EfficientNetB0 for feature extraction from images and an LSTM-based decoder for generating captions. The model is trained on the Flickr30K dataset, which consists of images and their corresponding captions.

## Key Components

- **Image Feature Extraction**: EfficientNetB0 is used to extract image features.
- **Text Preprocessing**: Captions are preprocessed by cleaning and tokenizing text.
- **Model Architecture**: The model combines image features from EfficientNetB0 with text sequences processed through an LSTM-based decoder.
- **Custom Data Generator**: A custom data generator is implemented to handle large datasets and perform batching efficiently.

## Installation

### Requirements

- Python 3.x
- TensorFlow >= 2.0
- NumPy
- Pandas
- Matplotlib
- Seaborn
- tqdm
- PIL

You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the **Flickr30K dataset**, which contains 31,000 images with associated captions. The images are located in the `input/flickr30k_images/Images` directory, and the captions are in the `input/flickr30k_images/results.csv` file.

### Dataset Structure

- **flickr30k_images/Images**: Folder containing the images.
- **results.csv**: CSV file containing the image names and corresponding captions. The captions are pre-processed for training the model.

## Model Architecture

### Image Feature Extraction

EfficientNetB0 is used to extract image features. The model is pre-trained on ImageNet, and the top layers are excluded to extract feature maps from the image.

### Text Generation (LSTM)

The captions are tokenized and processed using an LSTM model. The model generates one word at a time based on the image features.

### Custom Data Generator

The data generator handles batching and shuffling of data efficiently. It processes both image features and tokenized captions and prepares them for training.

## Training

To train the model, you need to run the script below:

```python
# Preprocess the data
data = pd.read_csv("input/flickr30k_images/results.csv", sep='|')
data = text_preprocessing(data)

# Initialize the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['comment'].tolist())

# Prepare data generators
train_data = CustomDataGenerator(df=train, X_col='image_name', y_col='comment', batch_size=64, tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_length, features=features)
val_data = CustomDataGenerator(df=test, X_col='image_name', y_col='comment', batch_size=64, tokenizer=tokenizer, vocab_size=vocab_size, max_length=max_length, features=features)

# Model training
model.fit(train_data, validation_data=val_data, epochs=20)
```


## Notes

    Data Preprocessing: Captions are tokenized and cleaned (e.g., removing special characters, converting to lowercase).
    Image Preprocessing: Images are resized to 224x224 pixels for EfficientNetB0.
    GPU Usage: The code is optimized for running on GPUs. Ensure that your environment has GPU support for better performance.

## Files

    tokenizer.pkl: Saved tokenizer for text processing.
    features_efficientnetb0.pkl: Extracted features from EfficientNetB0, saved to avoid reprocessing.
    model_checkpoint.h5: Saved model checkpoint after training.
    generate_caption.py: A script for generating captions using the trained model.

## Conclusion

This project demonstrates how to combine EfficientNetB0 for image feature extraction and an LSTM for text generation, resulting in an image captioning model capable of generating captions for unseen images.