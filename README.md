# Text Generation using LSTM Neural Network

This project demonstrates a simple text generation model using Long Short-Term Memory (LSTM) in TensorFlow. The model is trained on Shakespeare's text to predict the next character based on the input sequence. The trained model can generate human-like text given a starting string, using a specified temperature to control the creativity and randomness of the generated text.

## Requirements

- Python 3.x
- TensorFlow 2.x
- NumPy

You can install the necessary Python libraries using the following:

```bash
pip install tensorflow numpy
```

## Data Source

The dataset used for training is the works of William Shakespeare, which is publicly available through TensorFlow's `get_file` function. The text is processed and converted into a format suitable for training the model.

## File Description

- **shakespeare.txt**: The text file containing the works of Shakespeare, which is used as the source of the training data.

## Code Overview

### 1. Data Preprocessing

- The script first downloads and reads the Shakespeare text.
- The text is converted to lowercase to improve accuracy during training.
- A subset of the text (characters 300,000 to 800,000) is used for model training.
- Unique characters are extracted, and dictionaries (`char_to_num` and `num_to_char`) are created to map characters to numbers and vice versa.

### 2. Model Architecture

- The model is a simple LSTM neural network that predicts the next character in a sequence.
- It uses one LSTM layer with 128 units and a Dense output layer with softmax activation to predict the next character in the sequence.

### 3. Model Training

- The training process involves converting the input text into numerical data (one-hot encoding) and training the model on this data.
- The model is compiled with the categorical cross-entropy loss function and RMSprop optimizer.
- Once trained, the model is saved for future use.

### 4. Text Generation

- The trained model is loaded using `tf.keras.models.load_model`.
- The `generate_text` function generates text of a given length based on a random starting point.
- The temperature parameter is used to control the randomness of the generated text. Lower temperatures lead to more deterministic output, while higher temperatures make the output more creative.

### 5. Example Runs

The code generates text based on different temperature values:

- **Temperature 0.4**: The text is more predictable and consistent.
- **Temperature 0.6**: The text has a bit more variety, but still relatively coherent.
- **Temperature 0.8**: The text becomes more random and creative.

## Usage

### Running the Model

After training the model and saving it as `TextGenAI.keras`, you can load the model and generate text using the following command:

```python
# Generate text with a given length and temperature
print(generate_text(300, 0.4))  # Generates 300 characters with temperature 0.4
print(generate_text(300, 0.6))  # Generates 300 characters with temperature 0.6
print(generate_text(300, 0.8))  # Generates 300 characters with temperature 0.8
```

### Sample Output

When running the script, you will get output similar to the following:

```
==========================RUN 1 LEN 300, TEMP 0.4 ==========================
<Generated text with temperature 0.4>

==========================LEN 300, TEMP 0.6 ==========================
<Generated text with temperature 0.6>

==========================RUN 1 LEN 300, TEMP 0.8 ==========================
<Generated text with temperature 0.8>
```

## Saving and Loading the Model

The trained model can be saved with:

```python
model.save('TextGenAI.keras')
```

To load the saved model later, use:

```python
model = tf.keras.models.load_model('TextGenAI.keras')
```

## Notes

- The quality of text generation is highly dependent on the quality of training data and the complexity of the model.
- The temperature parameter significantly affects the output. A higher temperature results in more diverse but potentially less coherent text.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to modify the code and experiment with different text datasets, sequence lengths, and temperature values to observe how the model behaves under different conditions.