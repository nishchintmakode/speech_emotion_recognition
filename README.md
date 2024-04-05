# Speech Emotion Recognition using MLPClassifier

This project aims to recognize emotions from speech recordings using Multi-Layer Perceptron (MLP) Classifier. The dataset utilized is the **RAVDESS dataset** containing 1439 audio files with various emotions.

## Overview

Speech Emotion Recognition (SER) is a challenging task due to the complexity and variability of human speech. This project focuses on utilizing machine learning techniques to automatically classify the emotional content of speech recordings.

## Features

- **Audio Processing**: Audio files are loaded and processed using various libraries including librosa and scipy.
- **Feature Extraction**: Features such as Mel-frequency cepstral coefficients (MFCC), chroma, and mel-spectrogram are extracted to represent the emotional content of speech.
- **Model Training**: An MLP Classifier is trained on the extracted features to classify emotions.
- **Model Deployment**: The trained model is saved to be used for real-time predictions.

## Requirements

To run this project, ensure you have the following libraries installed:

- pandas
- numpy
- matplotlib
- scipy
- librosa
- noisereduce
- python_speech_features
- scikit-learn
- tensorflow

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/<username>/speech-emotion-recognition.git
   ```

2. Navigate to the project directory:

   ```bash
   cd speech-emotion-recognition
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the script to train the model:

   ```bash
   python train_model.py
   ```

5. After training, you can use the model for prediction:

   ```bash
   python predict.py <path_to_audio_file>
   ```

## Example

```bash
python predict.py path/to/audio/file.wav
```

## References

- [RAVDESS dataset](https://zenodo.org/record/1188976#.Yk0Y6JNKhTZ)
- [LibROSA documentation](https://librosa.org/doc/main/index.html)
- [scikit-learn documentation](https://scikit-learn.org/stable/documentation.html)

For more details and contributions, please refer to the [GitHub repository](https://github.com/nishchintmakode/speech-emotion-recognition).
