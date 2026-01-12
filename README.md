# DeBERTa Sentiment Analysis on Android with ExecuTorch

This is a sample Android application that demonstrates how to run a quantized DeBERTa model for sentiment analysis directly on-device using PyTorch ExecuTorch.

The app takes a text sentence from the user, tokenizes it using a custom Kotlin WordPiece tokenizer, and feeds it into a weight-quantized DeBERTa model to predict whether the sentiment is "Positive" or "Negative".

## Demo

*Click [here](https://youtu.be/vSnqgWuUc3M) for the demo, you may view up to 2:30 for this project and the second half [for this repository]()*

## Features

- **On-Device Inference**: All computation happens locally on the device, requiring no internet connection.
- **ExecuTorch Integration**: Uses the `executorch-android` library to load and run the included `.pte` model.
- **Custom Kotlin Tokenizer**: Includes a from-scratch implementation of a WordPiece tokenizer that uses the included `vocab.txt`.
- **Dynamic Shape Model**: The included model supports dynamic input shapes to handle sentences of varying lengths.
- **Quantization**: The included model is quantized using `int8_weight_only` to reduce its size.

## How to Build and Run

All necessary model files (`.pte`) and vocabulary files (`vocab.txt`) are already included in the `app/src/main/assets` directory. You just need to clone the repository and run it.

### Prerequisites

- Android Studio (latest stable version recommended)
- A physical Android device or a compatible emulator.

### Steps

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/lee1613/Android-LLM-Deployment
    cd Android-LLM-Deployment
    ```

2.  **Open in Android Studio**
    -   Open Android Studio.
    -   Select "Open" and navigate to the cloned `Android-LLM-Deployment` folder.

3.  **Build and Run**
    -   Let Gradle sync all the project dependencies.
    -   Click the **Run 'app'** button (the green play icon) to build and install the application on your device or emulator.
    -   Type a sentence and press "Predict" to see the result!
