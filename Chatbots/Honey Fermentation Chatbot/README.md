# Honey Fermentation Chatbot - User Guide

This document acts as a user guide for the chatbot designed to predict the Specialty Coffee Association (SCA) score for the honey fermentation process. Utilizing a predictive model, this chatbot estimates the SCA score based on several input parameters associated with the fermentation and drying stages. This guide offers step-by-step instructions on how to interact with the chatbot and enter the required data to obtain precise predictions.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Using the Chatbot](#using-the-chatbot)
4. [Model Features](#model-features)

---

## Introduction

The predictive model for honey fermentation aims to estimate the SCA score of coffee beans based on specific conditions during the fermentation and drying stages. This guide will walk you through setting up and using the chatbot to generate predictions, emphasizing key features and providing example predictions.

---

## Setup

### 1. Installing Necessary Libraries

Ensure you have the necessary libraries installed. This includes standard data manipulation libraries like `pandas` and `numpy`, visualization libraries such as `matplotlib` and `seaborn`, and machine learning libraries like `scikit-learn` and `xgboost`.

You can install these libraries using `pip`. Open your terminal or command prompt and run the following commands:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Loading the Model
Make sure the trained model is saved before running the chatbot. The model should be saved as **xgb_honey_model.json**.

## Using the Chatbot
The chatbot interacts with users to input relevant data and predict the SCA score based on the trained model.

### 1. Loading the Chatbot Script
The chatbot script should include the following code:

```python
import xgboost as xgb
import pandas as pd

# Load the saved model
# Initialize the XGBRegressor instance
xgb_honey_model = xgb.XGBRegressor()
xgb_honey_model.load_model('xgb_honey_model.json')

def get_user_input():
    """Prompt the user for input data"""
    input_data = {}
    input_data['Average Temp'] = float(input("Enter the average temperature: "))
    input_data['Average PH'] = float(input("Enter the average pH: "))
    input_data['Average Brix'] = float(input("Enter the average Brix level: "))
    input_data['Average Humidity'] = int(input("Enter the average humidity: "))
    input_data['Average Drying Temp'] = float(input("Enter the average drying temperature: "))
    input_data['Average Drying Humidity'] = float(input("Enter the average drying humidity: "))
    input_data['Barrel Open in (Hour)'] = int(input("Enter the number of hours the barrel is open: "))
    input_data['Barrel Close in (Hour)'] = int(input("Enter the number of hours the barrel is closed: "))
    input_data['Fermentation Duration (hours)'] = int(input("Enter the fermentation duration (in hours): "))
    input_data['Drying Fermentation Duration (hours)'] = int(input("Enter the drying duration (in hours): "))
    input_data['Sunny'] = int(input("Enter the number of sunny days: "))
    input_data['Cloudy'] = int(input("Enter the number of cloudy days: "))
    input_data['Rainy'] = int(input("Enter the number of rainy days: "))
    input_data['Overcast'] = int(input("Enter the number of overcast days: "))
    input_data['Light Rain'] = int(input("Enter the number of light rain days: "))
    input_data['Heavy Rain'] = int(input("Enter the number of heavy rain days: "))
    input_data['Coffee Variety_Special'] = bool(input("Is the coffee variety special? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_Molasses'] = bool(input("Are Additives for Fermentation Molasses? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_None'] = bool(input("Are there no Additives for Fermentation? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_Sugar'] = bool(input("Are Additives for Fermentation Sugar? (yes/no): ").strip().lower() == 'yes')
    input_data['Additives for fermentation_Yeast'] = bool(input("Are Additives for Fermentation Yeast? (yes/no): ").strip().lower() == 'yes')
    input_data['Pre-fermentation for Honey_Yes'] = bool(input("Is there pre-fermentation? (yes/no): ").strip().lower() == 'yes')

    if input_data['Pre-fermentation for Honey_Yes']:
        input_data['Pre-fermentation Details (Hour) for Honey'] = int(input("Enter the pre-fermentation details (in hours) for honey: "))
    else:
        input_data['Pre-fermentation Details (Hour) for Honey'] = 0

    # Ensure the columns match the model's expected order
    expected_columns = ['Average Temp', 'Average PH', 'Average Brix', 'Average Humidity', 'Average Drying Temp', 
                        'Average Drying Humidity', 'Barrel Open in (Hour)', 'Barrel Close in (Hour)', 
                        'Pre-fermentation Details (Hour) for Honey', 'Fermentation Duration (hours)', 
                        'Drying Fermentation Duration (hours)', 'Sunny', 'Cloudy', 'Rainy', 'Overcast', 
                        'Light Rain', 'Heavy Rain', 'Coffee Variety_Special', 'Additives for fermentation_Molasses', 
                        'Additives for fermentation_None', 'Additives for fermentation_Sugar', 
                        'Additives for fermentation_Yeast', 'Pre-fermentation for Honey_Yes']

    input_data = {k: input_data[k] for k in expected_columns}

    return input_data

def predict_sca_score(input_data):
    """Predict SCA score based on user input"""
    input_df = pd.DataFrame([input_data])
    prediction = xgb_honey_model.predict(input_df)
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

The following variables are crucial inputs that help the model predict the SCA score for coffee beans processed through honey fermentation.

## Important Variables

- **Average Temp:** The average temperature during fermentation.
- **Average PH:** The average pH level during fermentation.
- **Average Brix:** The average Brix level (sugar content) during fermentation.
- **Average Humidity:** The average humidity level during fermentation.
- **Average Drying Temp:** The average temperature during drying.
- **Average Drying Humidity:** The average humidity level during drying.
- **Barrel Open in (Hour):** The number of hours the barrel is open during fermentation.
- **Barrel Close in (Hour):** The number of hours the barrel is closed during fermentation.
- **Fermentation Duration (hours):** The duration of fermentation in hours.
- **Drying Fermentation Duration (hours):** The duration of drying in hours.
- **Sunny, Cloudy, Rainy, Overcast, Light Rain, Heavy Rain:** The number of days with these weather conditions during fermentation.
- **Coffee Variety_Special:** Whether the coffee variety is special.
- **Additives for fermentation_Molasses, None, Sugar, Yeast:** The additives used during fermentation.
- **Pre-fermentation for Honey_Yes:** Whether pre-fermentation is used.
- **Pre-fermentation Details (Hour) for Honey:** The duration of pre-fermentation in hours (if used).



