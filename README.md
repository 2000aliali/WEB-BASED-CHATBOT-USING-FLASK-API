# Web-Based Chatbot Using Flask API

This Python web-based project with source code demonstrates how to build a chatbot using deep learning and Flask techniques. The chatbot is trained on a dataset containing categories (intents), patterns, and responses. It utilizes an artificial neural network (ANN) to classify the user's message category and provides a random response from a list of responses.

## Technologies Used
- Python
- Flask
- NLTK (Natural Language Toolkit)
- Keras

## How to Make a Chatbot in Python?

Now, let's explore how to build a chatbot using Flask framework:

### File Structure

Here's an overview of the file structure and the types of files we'll be creating:

- `data.json`: The data file which has predefined patterns and responses.
- `training.py`: In this Python file, we wrote a script to build the model and train our chatbot.
- `Texts.pkl`: This is a pickle file in which we store the words as a Python object using NLTK, containing a list of our vocabulary.
- `Labels.pkl`: The classes pickle file contains the list of categories (labels).
- `model.h5`: This is the trained model that contains information about the model and has weights of the neurons.
- `app.py`: This is the Flask Python script in which we implemented a web-based GUI for our chatbot. Users can easily interact with the bot.

### 5 Steps to Create a Chatbot in Flask from Scratch

Here are the 5 key steps to create a chatbot in Flask:

1. **Import and Load the Data File**: Load the data from the `data.json` file.

2. **Preprocess Data**: Prepare the data for training.

3. **Split Data**: Divide the data into training and testing sets.

4. **Build the ANN Model Using Keras**: Create an Artificial Neural Network (ANN) model using Keras.

5. **Predict the Outcomes**: Train the model and predict responses.

6. **Deploy the Model in the Flask App**: Implement the chatbot in a Flask-based web GUI.

## Getting Started

To get started, make sure to install the required dependencies using `pip`:

```bash
pip install tensorflow
pip install keras
pip install pickle
pip install nltk
pip install flask

