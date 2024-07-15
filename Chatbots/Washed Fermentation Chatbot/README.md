# Washed Fermentation Chatbot - User Guide

This document serves as a user guide for the chatbot developed to predict the Specialty Coffee Association (SCA) score for the washed fermentation process. This chatbot uses a predictive model to forecast the SCA score based on various input parameters related to the fermentation and drying processes. This guide provides step-by-step instructions on how to interact with the chatbot and input the necessary data to receive accurate predictions.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Using the Chatbot](#using-the-chatbot)
4. [Model Features](#model-features)


---

## Introduction

The predictive model for washed fermentation aims to forecast the SCA score of coffee beans based on specific conditions during the fermentation and drying processes. This guide will walk you through setting up and using the chatbot to make predictions, highlighting the key features and providing example predictions.

---

## Setup

### 1. Installing Necessary Libraries

Ensure you have the necessary libraries installed. This includes standard data manipulation libraries like `pandas` and `numpy`, visualization libraries such as `matplotlib` and `seaborn`, and machine learning libraries like `scikit-learn` and `xgboost`.

You can install these libraries using `pip`. Open your terminal or command prompt and run the following commands:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

### 2. Loading the Model
Make sure the trained model is saved before running the chatbot. The model should be saved as **xgb_washed_model.pkl**.

## Using the Chatbot
The chatbot interacts with users to input relevant data and predict the SCA score based on the trained model.

### 1. Loading the Chatbot Script
The chatbot script should include the following code:

```python
import joblib
import pandas as pd

# Load the saved model
xgb_washed_model = joblib.load('xgb_washed_model.pkl')

def get_user_input():
    """Prompt the user for input data"""
    input_data = {}
    input_data['Average Temp'] = float(input("Enter Average Temp: "))
    input_data['Average PH'] = float(input("Enter Average PH: "))
    input_data['Average Brix'] = float(input("Enter Average Brix: "))
    input_data['Average Humidity'] = float(input("Enter Average Humidity: "))
    input_data['Average Drying Temp'] = float(input("Enter Average Drying Temp: "))
    input_data['Average Drying Humidity'] = float(input("Enter Average Drying Humidity: "))
    input_data['Fermentation Duration (hours)'] = int(input("Enter Fermentation Duration (hours): "))
    input_data['Drying Fermentation Duration (hours)'] = int(input("Enter Drying Fermentation Duration (hours): "))
    input_data['Sunny'] = int(input("Enter number of Sunny days: "))
    input_data['Cloudy'] = int(input("Enter number of Cloudy days: "))
    input_data['Rainy'] = int(input("Enter number of Rainy days: "))
    input_data['Partly Cloudy'] = int(input("Enter number of Partly Cloudy days: "))
    input_data['Overcast'] = int(input("Enter number of Overcast days: "))
    input_data['Light Rain'] = int(input("Enter number of Light Rain days: "))
    input_data['Heavy Rain'] = int(input("Enter number of Heavy Rain days: "))
    input_data['Coffee Variety_Special'] = bool(input("Is the Coffee Variety Special? (yes/no): ").strip().lower() == 'yes')
    input_data['Type of Water Used_Spring Water'] = bool(input("Is the Type of Water Used Spring Water? (yes/no): ").strip().lower() == 'yes')
    input_data['Type of Water Used_Well Water'] = bool(input("Is the Type of Water Used Well Water? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_Molasses'] = bool(input("Are Additives for Fermentation Molasses? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_None'] = bool(input("Are there no Additives for Fermentation? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_Sugar'] = bool(input("Are Additives for Fermentation Sugar? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_Yeast'] = bool(input("Are Additives for Fermentation Yeast? (yes/no): ").strip().lower() == 'yes')
    
    return input_data

def predict_sca_score(input_data):
    """Predict SCA score based on user input"""
    input_df = pd.DataFrame([input_data])
    prediction = xgb_washed_model.predict(input_df)
    return prediction[0]

def chatbot():
    print("Welcome to the SCA Score Prediction Chatbot!")
    while True:
        user_input = get_user_input()
        sca_score = predict_sca_score(user_input)
        print(f"Predicted SCA Score: {sca_score}")
        
        cont = input("Would you like to input another set of data? (yes/no): ").strip().lower()
        if cont != 'yes':
            print("Thank you for using the SCA Score Prediction Chatbot. Goodbye!")
            break

if __name__ == "__main__":
    chatbot()
```

### 2. Running the Chatbot

Run the script in your Python environment. The chatbot will prompt you to enter the necessary data, predict the SCA score, and ask if you want to input another set of data (A demonstration video will be provided in the folder to showcase how to use the chatbot).

## Model Features

### Important Variables

The following variables are crucial inputs that help the model predict the SCA score for coffee beans processed through washed fermentation.

- **Average Temp:** The average temperature during fermentation.
- **Average PH:** The average pH level during fermentation.
- **Average Brix:** The average Brix level (sugar content) during fermentation.
- **Average Humidity:** The average humidity level during fermentation.
- **Average Drying Temp:** The average temperature during drying.
- **Average Drying Humidity:** The average humidity level during drying.
- **Fermentation Duration (hours):** The duration of fermentation in hours.
- **Drying Fermentation Duration (hours):** The duration of drying in hours.
- **Sunny, Cloudy, Rainy, Partly Cloudy, Overcast, Light Rain, Heavy Rain:** The number of days with these weather conditions during fermentation.
- **Coffee Variety_Special:** Whether the coffee variety is special.
- **Type of Water Used_Spring Water, Type of Water Used_Well Water:** The type of water used in the process.
- **Additives for fermentation_Molasses, None, Sugar, Yeast:** The additives used during fermentation.


