# Overview of Coffee Fermentation Predictive Chatbots

This document provides an overview of two chatbots developed to enhance the Specialty Coffee Association (SCA) score for coffee beans processed through two different fermentation methods: washed fermentation and honey (anaerobic) fermentation. Each chatbot utilizes a predictive model that leverages synthetic historical data and machine learning techniques to forecast optimal conditions for fermentation and drying, thus improving coffee quality.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Washed Fermentation Chatbot](#washed-fermentation-chatbot)
3. [Honey Fermentation Chatbot](#honey-fermentation-chatbot)
4. [Necessity for Two Chatbots](#necessity-for-two-chatbots)
5. [Common Features](#common-features)

---

## Introduction

The two chatbots are designed to estimate the SCA score for coffee beans based on specific conditions during the fermentation and drying processes. The chatbots facilitate interaction with the predictive models, allowing users to input necessary data and receive accurate predictions. This guide provides a brief overview of the chatbots, highlighting their features and the reasons for developing separate chatbots for each fermentation method.

## Before running the chatbot, please make sure you have followed the YouTube Video to download Anaconda, and be able to use Jupyter Notebook on you laptop. 
(YouTube Video link: https://www.youtube.com/watch?v=drbaFALFKDg).

## Code Demonstration Videos
There are two code demonstration videos provided, one for each chatbot. These videos are located in their respective folders and are designed to assist users in understanding how to effectively use the chatbots.

---

## Washed Fermentation Chatbot

### Overview
The washed fermentation chatbot uses the predictive model to estimate the SCA score for coffee beans processed through the washed fermentation method. Users interact with the chatbot by inputting specific variables related to the fermentation and drying processes. The chatbot then provides an accurate prediction of the SCA score based on the input data. 

### Important Variables
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

---

## Honey Fermentation Chatbot

### Overview
The honey fermentation chatbot uses the predictive model to estimate the SCA score for coffee beans processed through the honey fermentation method. Users interact with the chatbot by inputting specific variables related to the fermentation and drying processes. The chatbot then provides an accurate prediction of the SCA score based on the input data.

### Important Variables
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

---

## Necessity for Two Chatbots

The necessity for having two separate chatbots arises from the distinct differences in the fermentation processes and the variables that impact coffee quality.

- **Washed Fermentation:** This method involves soaking the coffee cherries in water to remove the outer fruit layer before drying. Key variables such as the type of water used and the drying conditions significantly impact the SCA score.
- **Honey Fermentation:** This method involves removing the outer skin of the coffee cherry but leaving some or all of the sticky fruit layer on the beans during drying. This process can be further influenced by whether the beans undergo pre-fermentation and the specific conditions within the barrel.

By having two separate chatbots, each model can be tailored to the specific variables and conditions that affect the respective fermentation processes, leading to more accurate predictions and better optimization of the coffee quality.

---

## Common Features

Both chatbots share the following features:
- **User-Friendly Interface:** Simple chatbot interface for predicting SCA scores based on user inputs, enhancing usability for clients.
- **Accurate Predictions:** Leverage machine learning models to provide accurate SCA score predictions based on input variables.
- **Scalability:** Can predict SCA scores for new data, making them scalable solutions for continuous quality improvement.

For detailed instructions on how to use each chatbot, please refer to the individual folder provided with each Jupyter Notebook. A demonstration video will be provided in these folders to showcase how to use the chatbots effectively.
