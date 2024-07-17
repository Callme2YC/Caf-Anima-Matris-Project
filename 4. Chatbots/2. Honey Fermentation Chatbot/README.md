# Honey Fermentation Chatbot - User Guide

This document serves as a user guide for the chatbot created to predict the Specialty Coffee Association (SCA) score for the honey fermentation process. The chatbot leverages a predictive model to estimate the SCA score based on various input parameters related to the fermentation and drying stages. This guide provides detailed instructions on how to interact with the chatbot and input the necessary data to receive accurate predictions.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Using the Chatbot](#using-the-chatbot)
4. [Model Features](#model-features)
5. [Notifications](#notifications)
   
---

## Introduction

The predictive model for honey fermentation aims to estimate the SCA score of coffee beans based on specific conditions during the fermentation and drying stages. This guide will walk you through setting up and using the chatbot to generate predictions, emphasizing key features and providing example predictions.

---

## Setup

### 1. Download Necessary Files

Before proceeding with the installation of libraries, ensure you have downloaded the following necessary files:

1. **Washed Fermentation Chatbot Notebook:** Download the Jupyter Notebook file named `Honey Fermentation Chatbot.ipynb`.
2. **Gradient Boosting Model:** Download the model file named `gb_honey_model.pkl`.
3. **Code Demo:** Download the demo video file named `z_Code Demo - Honey Fermentation Chatbot.mp4` and visit it before running the chatbot.

Make sure to keep the filenames as specified above to avoid any issues when running the notebook. Do not rename the files to ensure compatibility with the code.


### 2. Installing Necessary Libraries

Ensure you have the necessary libraries installed. This includes standard data manipulation libraries like `pandas` and `numpy`,  and machine learning libraries like `scikit-learn`.

You can install these libraries using `pip`. Open your terminal or command prompt and run the following commands:

```sh
!pip install pandas numpy scikit-learn
```

### 2. Loading the Model
Make sure the trained model is saved before running the chatbot. The model should be saved as **gb_honey_model.pkl**.

## Using the Chatbot
The chatbot interacts with users to input relevant data and predict the SCA score based on the trained model.

### 1. Loading the Necessary Libraries

```python
import pandas as pd
import joblib
```

### 2. Loading the Chatbot Script
The chatbot script should include the following code:

```python
# Load the saved model
gb_honey_model = joblib.load('gb_honey_model.pkl')

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
    prediction = gb_honey_model.predict(input_df)
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

### 3. Running the Chatbot

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

## Notifications

1. **Numerical Inputs:**
   - Ensure that the numerical inputs fall within the following reasonable ranges:
     - **Average Temp:** 18°C to 30°C
     - **Average PH:** 4.0 to 6.0
     - **Average Brix:** 15 to 25
     - **Average Humidity:** 60% to 80%
     - **Average Drying Temp:** 20°C to 35°C
     - **Average Drying Humidity:** 50% to 70%
     - **Fermentation Duration (hours):** 48 to 120 hours
     - **Drying Fermentation Duration (hours):** 72 to 168 hours
   - Inputs should be numerical only, with no units required.
   - Inputs outside these ranges may affect the model's accuracy.

2. **Weather Conditions:**
   - The fermentation process generally lasts 3 to 5 days. Ensure the total number of days with specified weather conditions is reasonable within this timeframe.
   - **Example:** If the fermentation lasts 4 days, you might specify 2 sunny days, 1 cloudy day, and 1 rainy day, while other weather conditions should be 0. Ensure the total number of days specified does not exceed 4.

3. **Barrel Open in (Hour):**
   - Honey fermentation is conducted in a sealed environment (i.e., in the barrel). However, it is occasionally necessary to open the barrel to mix the beans, particularly when the temperature is too high. Opening the barrel can help reduce the temperature. For this variable, a range of 0 to 3 hours has been assigned during the model development stage.

4. **Barrel Close in (Hour):**
   - This parameter should generally be the same as the fermentation duration hour. Ensure this rule is followed.

5. **Type of Water Used:**
   - There is no questino for the "Typy of Water User", since honey fermentation do not need water in this process.

6. **Pre-fermentation for Honey:**
   - If "Yes" is entered for "Whether pre-fermentation is used," the chatbot will prompt you to input the duration of the pre-fermentation in hours. If "No" is entered, this question will not be presented.

7. **Additives for Fermentation:**
   - Only one additive should be selected and used during the fermentation process.

Please adhere to these guidelines to ensure accurate and reliable predictions from the model.


