# Web-Based Chatbot Using Flask API

This Python web-based project with source code demonstrates how to build a chatbot using deep learning and Flask techniques. The chatbot is trained on a dataset containing categories (intents), patterns, and responses. It utilizes an artificial neural network (ANN) to classify the user's message category and provides a random response from a list of responses.

## Technologies Used
- Python
- Flask
- NLTK (Natural Language Toolkit)
- Keras

## Project Overview

The heart of this project is the `data.json` file, which contains predefined patterns and responses. We use this data to train our chatbot. The project involves using Python, Keras, and Natural Language Processing (NLTK) along with some additional modules.

To get started, make sure to install the required dependencies using `pip`:

```bash
pip install tensorflow
pip install keras
pip install pickle
pip install nltk
pip install flask
```


##            How to Make Chatbot in Python?

Now we are going to build the chatbot using Flask framework but first, let us see the file structure and the type of files we will be creating:

data.json – The data file which has predefined patterns and responses.
trainning.py – In this Python file, we wrote a script to build the model and train our chatbot.
Texts.pkl – This is a pickle file in which we store the words Python object using Nltk that contains a list of our vocabulary.
Labels.pkl – The classes pickle file contains the list of categories(Labels).
model.h5 – This is the trained model that contains information about the model and has weights of the neurons.
app.py – This is the flask Python script in which we implemented web-based GUI for our chatbot. Users can easily interact with the bot.
Here are the 5 steps to create a chatbot in Flask from scratch:

- 1.Import and load the data file
- 2.Preprocess data
- 3.split the data into training and test
-4.Build the ANN model using keras
-5.Predict the outcomes
-6.Deploy the model in the Flask app
