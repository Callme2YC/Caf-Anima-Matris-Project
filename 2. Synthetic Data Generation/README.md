# Café Anima Matris Synthetic Data Generation

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Data Generation Process](#data-generation-process)
4. [Detailed Workflow](#detailed-workflow)
5. [Rationale for Each Column](#rationale-for-each-column)
6. [Conclusion](#conclusion)

## Introduction

This guide provides a detailed overview for future developers on how the code for generating synthetic data for the Café Anima Matris project works. This document demonstrates how data was generated based on the data collection template developed for clients. The columns built are intended to prepare for current and future model training.

## Setup

### Installing Necessary Packages

Ensure you have the following Python packages installed:
- pandas
- Faker
- numpy

You can install these packages using pip:

```bash
pip install pandas faker numpy
```

## Data Generation Process

### Objective
The objective of this code is to generate a synthetic dataset that simulates real-world conditions of coffee fermentation and drying processes. This dataset is crucial for modeling and analysis purposes.

### Process Flow
1. **Initialization:** Set up the environment and initialize necessary libraries for data generation.
2. **Define Data Parameters and Rules:** Specify the range of dates, batch IDs, and other metrics to be collected.
3. **Data Generation:** Generate 5000 rows of synthetic data adhering to the defined rules.
4. **Data Storage:** Save the generated data as a CSV file.
5. **Verification and Validation:** Ensure the generated data adheres to the defined rules and ranges.

## Detailed Workflow

### 1. Import necessary libraries:

```python
import pandas as pd
from faker import Faker
import numpy as np
import random
from datetime import datetime, timedelta
```

### 2. Define Data Parameters and Rules
   
Set up the environment:

```python
# Initialize Faker
fake = Faker()

# Function to generate random date and time within a range
def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end - start).total_seconds())))

# Realistic weather conditions for Colombia
weather_conditions_colombia = [
    "Sunny", "Cloudy", "Rainy", "Partly Cloudy", "Overcast", "Light Rain", "Heavy Rain"
]
weather_probs = [0.2, 0.3, 0.1, 0.1, 0.1, 0.15, 0.05]  # Probabilities for each weather condition

# Types of water commonly used in Colombia for coffee fermentation
types_of_water = ["Spring Water", "Well Water", "Filtered Water"]
water_probs = [0.8, 0.1, 0.1]  # Probabilities for each type of water

# Additives for fermentation
additives = ["None", "Molasses", "Yeast", "Honey", "Sugar"]
additives_probs = [0.8, 0.05, 0.05, 0.05, 0.05]  # Probabilities for each additive

# Coffee varieties
coffee_varieties = ["Special", "Normal"]
coffee_variety_probs = [0.8, 0.2]  # Mostly "Special"

# Fermentation methods
fermentation_methods = ["Washed", "Honey"]
fermentation_method_probs = [0.5, 0.5]  # Equal distribution
```

### 3. Generate Synthetic Data

Generate 5000 rows of synthetic data:

```python
# Create a DataFrame with 5000 rows of fake data
data = []
num_records = 5000

# Generate unique Batch IDs and Batch Names
batch_ids = [f"Batch {i+1}" for i in range(num_records)]
batch_names = [f"Batch {i+1}" for i in range(num_records)]

# Define the date range for the last three years
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)

# Generate realistic distributions for data
for i in range(num_records):
    fermentation_method = np.random.choice(fermentation_methods, p=fermentation_method_probs)
    fermentation_start = fake.date_time_between(start_date=start_date, end_date=end_date)
    fermentation_end = fermentation_start + timedelta(days=random.randint(3, 5), hours=random.randint(0, 24))
    drying_start = fermentation_end + timedelta(hours=random.randint(1, 12))
    drying_end = drying_start + timedelta(days=random.randint(3, 5), hours=random.randint(0, 24))
    
    # Temperature (normal distribution)
    fermentation_temp = np.random.normal(23, 3)  # Normal distribution with mean 23 and std dev 3
    fermentation_temp = np.clip(fermentation_temp, 18.0, 28.0)  # Ensure values are within the range
    
    drying_temp = np.random.normal(23, 3)
    drying_temp = np.clip(drying_temp, 18.0, 28.0)
    
    # Humidity (normal distribution correlated with temperature)
    fermentation_humidity = np.random.normal(75, 10) + (fermentation_temp - 23) * 0.5  # Base 75% with temperature correlation
    fermentation_humidity = np.clip(fermentation_humidity, 60, 90)
    
    drying_humidity = np.random.normal(75, 10) + (drying_temp - 23) * 0.5
    drying_humidity = np.clip(drying_humidity, 60, 90)
    
    # pH Levels (normal distribution)
    ph_level = np.random.normal(4.5, 0.25)
    ph_level = np.clip(ph_level, 4.0, 5.0)
    
    # Fermentation Duration
    fermentation_duration_hours = (fermentation_end - fermentation_start).total_seconds() / 3600  # Duration in hours
    fermentation_duration_days = fermentation_duration_hours / 24  # Duration in days
    
    # SCA Score (separate calculations for Washed and Honey)
    base_sca_score = 85  # Base score
    
    if fermentation_method == 'Washed':
        temp_factor = (fermentation_temp - 23) * 0.2  # Moderate Temperature impact factor
        time_factor = (fermentation_duration_hours - 96) * 0.03  # Moderate Time impact factor
        ph_factor = (ph_level - 4.5) * 1  # Moderate pH level impact factor
        interaction_term = 0.005 * (fermentation_temp - 23) * (fermentation_duration_hours - 96)  # Interaction term
        humidity_factor = (fermentation_humidity - 75) * 0.09  # Moderate Humidity factor
        drying_temp_factor = (drying_temp - 23) * 0.1  # Moderate Drying Temp factor
        random_factor = np.random.normal(0, 0.5)  # Controlled random noise
        non_linear_term = np.log(fermentation_duration_hours) * 0.05  # Non-linear term
        sca_score = base_sca_score + temp_factor + time_factor + ph_factor + interaction_term + humidity_factor + drying_temp_factor + random_factor + non_linear_term
        sca_score = np.clip(sca_score, 80, 90)
    
    elif fermentation_method == 'Honey':
        temp_factor = (fermentation_temp - 23) * 0.35  # Reduced Temperature impact factor
        time_factor = (fermentation_duration_hours - 96) * 0.08  # Reduced Time impact factor
        ph_factor = (ph_level - 4.5) * 2  # Reduced pH level impact factor
        interaction_term = 0.01 * (fermentation_temp - 23) * (fermentation_duration_hours - 96)  # Interaction term
        humidity_factor = (fermentation_humidity - 75) * 0.09  # Reduced Humidity factor
        drying_temp_factor = (drying_temp - 23) * 0.15  # Reduced Drying Temp factor
        barrel_open_hours = random.randint(0, 3)  # Hours the barrel was open, each 3-4 hours within the range
        barrel_open_factor = barrel_open_hours * 0.5  # Significant factor for barrel open hours
        pre_fermentation_factor = pre_fermentation_details * 0.1 if pre_fermentation == "Yes" else 0  # Factor for pre-fermentation
        random_factor = np.random.normal(0, 1.0)  # Controlled noise for more variability
        sca_score = base_sca_score + temp_factor + time_factor + ph_factor + interaction_term + humidity_factor + drying_temp_factor + barrel_open_factor + pre_fermentation_factor + random_factor
        sca_score = np.clip(sca_score, 80, 90)
    
    # Brix Levels (beta distribution scaled to desired range)
    brix_levels = np.random.beta(2, 2) * 5 + 15  # Beta distribution scaled to range 15-20
    
    # Weather Conditions
    num_weather_conditions = random.randint(3, 5)
    fermentation_weather = ",".join(np.random.choice(weather_conditions_colombia, num_weather_conditions, p=weather_probs))
    
    coffee_variety = np.random.choice(coffee_varieties, p=coffee_variety_probs)
    pre_fermentation = np.random.choice(["Yes", "No"], p=[0.3, 0.7])  # More likely to be "No"
    pre_fermentation_details = random.randint(0, 24) if pre_fermentation == "Yes" else 0
    weather_conditions = np.random.choice(weather_conditions_colombia, p=weather_probs)
    type_of_water = np.random.choice(types_of_water, p=water_probs)
    additives_fermentation = np.random.choice(additives, p=additives_probs)
    
    # Record for barrels (only for Honey)
    barrel_open_hours = barrel_close_hours = ''
    if fermentation_method == 'Honey':
        barrel_open_hours = random.randint(0, 3)  # Hours the barrel was open, each 3-4 hours within the range
        barrel_close_hours = random.randint(60, 120) - barrel_open_hours  # Hours the barrel was closed
    
    # Fermentation records
    num_records_fermentation = random.randint(3, 5)
    fermentation_records_dates = [random_date(fermentation_start, fermentation_end) for _ in range(num_records_fermentation)]
    fermentation_records = ",".join([f"ferm_{batch_ids[i]}_{dt.isoformat()}" for dt in fermentation_records_dates])
    
    # Drying records
    num_records_drying = random.randint(3, 5)
    drying_records_dates = [random_date(drying_start, drying_end) for _ in range(num_records_drying)]
    drying_records = ",".join([f"dry_{batch_ids[i]}_{dt.isoformat()}" for dt in drying_records_dates])
    
    # Form submission dates
    form_submitted = random_date(fermentation_start, fermentation_start + timedelta(hours=12))
    form_modified = random_date(fermentation_end, fermentation_end + timedelta(hours=12))
    
    record = [
        batch_ids[i], batch_names[i], coffee_variety, 
        fermentation_end.strftime("%m/%d/%Y"), 
        fermentation_method, type_of_water if fermentation_method == "Washed" else '', 
        additives_fermentation, int(sca_score),
        fermentation_records,
        round(fermentation_temp, 1), 
        round(ph_level, 3), 
        round(brix_levels, 1), 
        int(fermentation_humidity),
        fermentation_weather, 
        fermentation_start.strftime("%m/%d/%Y %I:%M %p"), 
        fermentation_end.strftime("%m/%d/%Y %I:%M %p"),
        drying_records,
        drying_start.strftime("%m/%d/%Y %I:%M %p"), 
        drying_end.strftime("%m/%d/%Y %I:%M %p"), 
        form_submitted.strftime("%m/%d/%Y %I:%M %p"), form_modified.strftime("%m/%d/%Y %I:%M %p"), 
        round(drying_temp, 1), 
        round(drying_humidity, 3), 
        barrel_open_hours if fermentation_method == 'Honey' else '', 
        barrel_close_hours if fermentation_method == 'Honey' else '',
        pre_fermentation if fermentation_method == 'Honey' else '',
        pre_fermentation_details if fermentation_method == 'Honey' else ''
    ]

    data.append(record)

# Define column names
columns = [
    "Batch ID", "Batch Name", "Coffee Variety", "Harvest Date", "Fermentation Method", 
    "Type of Water Used", "Additives for fermentation", "SCA Score", "Fermentation Records", 
    "Average Temp", "Average PH", "Average Brix", "Average Humidity", "Fermentation Weather", 
    "Fermentation Start", "Fermentation End", "Drying Records", "Drying Start Date/Time", 
    "Drying End Date/Time", "Form Submitted Date/Time", "Form Modified Date/Time", 
    "Average Drying Temp", "Average Drying Humidity", "Barrel Open in (Hour)", "Barrel Close in (Hour)",
    "Pre-fermentation for Honey", "Pre-fermentation Details (Hour) for Honey"
]

# Create DataFrame
df = pd.DataFrame(data, columns=columns)

# Please replace the file path to fit your own work path
# Specify the file path
file_path = '/Users/xxx/Desktop/xxx/xxxx/fake_main_data.csv'

# Save to CSV
df.to_csv(file_path, index=False)

print(f"File saved to {file_path}")
```

## Rationale for Each Column

### Batch ID
- **Assumption:** Each batch must have a unique identifier.
- **Rule:** Generated using a combination of a sequential number and the fermentation process end timestamp.

### Batch Name
- **Assumption:** Each batch name is unique and sequential.
- **Rule:** Generated using a sequential number.

### Fermentation Start Date
- **Assumption:** Fermentation start dates must be within the last three years.
- **Rule:** Random date generated between 2021 and the current date.

### Fermentation End Date
- **Assumption:** Fermentation duration is between 3 to 5 days.
- **Rule:** Calculated by adding 3 to 5 days (plus up to 24 hours) to the fermentation start time.

### Drying Start Date
- **Assumption:** Drying starts within 1 to 12 hours after fermentation ends.
- **Rule:** Randomly generated time within 1 to 12 hours after the fermentation end time.

### Drying End Date
- **Assumption:** Drying duration is between 3 to 5 days.
- **Rule:** Calculated by adding 3 to 5 days (plus up to 24 hours) to the drying start time.

### Temperature
- **Assumption:** Temperature during the process is between 18.0°C and 28.0°C typical for Colombia's coffee regions.
- **Rule:** Randomly generated using a normal distribution with a mean of 23°C and a standard deviation of 3°C, clipped to the range 18.0°C to 28.0°C.

### Humidity
- **Assumption:** Humidity levels during the fermentation process are normally distributed with a mean of 75%, influenced by temperature, with a standard deviation of 10%.
- **Rule:** Generate humidity values using a normal distribution and add a slight correlation to temperature. Clip values within the range of 60% to 90%.

### pH Level
- **Assumption:** pH level during fermentation is between 4.0 and 5.0.
- **Rule:** Randomly generated using a normal distribution with a mean of 4.5 and a standard deviation of 0.25, clipped to the range 4.0 to 5.0.

### Pre-fermentation
- **Assumption:** 50% chance of having pre-fermentation, only applicable to the honey method.
- **Rule:** Randomly assigned "Yes" or "No" for the honey method only.

### Pre-fermentation Details
- **Assumption:** If pre-fermentation is "Yes," duration is between 0 to 24 hours.
- **Rule:** Randomly generated duration if pre-fermentation is "Yes"; otherwise set to 0. This is applicable only to the honey method.

### Weather Conditions
- **Assumption:** The weather conditions during the fermentation process follow typical patterns in Colombia.
- **Rule:** Generate weather conditions using predefined probabilities:
  - Sunny: 20%
  - Cloudy: 30%
  - Rainy: 10%
  - Partly Cloudy: 10%
  - Overcast: 10%
  - Light Rain: 15%
  - Heavy Rain: 5%

### Brix Levels
- **Assumption:** Brix levels during the process range between 15 and 20.
- **Rule:** Generate Brix levels using a beta distribution scaled to the desired range.

### Type of Water Used
- **Assumption:** The type of water used for fermentation is typically spring water, well water, or filtered water.
- **Rule:** Assign the type of water used with predefined probabilities:
  - Spring Water: 50%
  - Well Water: 20%
  - Filtered Water: 30%
- **Note:** This rule is only applicable to the washed method, as no water is used in the honey method.

### Additives for Fermentation
- **Assumption:** Additives are not commonly used, but when used, they include molasses, yeast, honey, and sugar.
- **Rule:** Assign additives with predefined probabilities:
  - None: 60%
  - Molasses: 10%
  - Yeast: 10%
  - Honey: 10%
  - Sugar: 10%

### SCA Score
- **Assumption:** The SCA score is influenced by temperature, fermentation duration, and pH levels with some added randomness.
- **Rule:** 
  - **Washed Method:** Calculate the SCA score using a base score of 85 adjusted by:
    - Temperature impact: (temperature - 23) * 0.2
    - Fermentation duration impact: (fermentation_duration - 96) * 0.03
    - pH level impact: (pH - 4.5) * 1
    - Interaction term: 0.005 * (temperature - 23) * (fermentation_duration - 96)
    - Humidity factor: (fermentation_humidity - 75) * 0.09
    - Drying temperature factor: (drying_temp - 23) * 0.1
    - Random noise: np.random.normal(0, 0.5)
    - Non-linear term: np.log(fermentation_duration) * 0.05
    - Clip the final score within the range of 80 to 90.
  - **Honey Method:** Calculate the SCA score using a base score of 85 adjusted by:
    - Temperature impact: (temperature - 23) * 0.35
    - Fermentation duration impact: (fermentation_duration - 96) * 0.08
    - pH level impact: (pH - 4.5) * 2
    - Interaction term: 0.01 * (temperature - 23) * (fermentation_duration - 96)
    - Humidity factor: (fermentation_humidity - 75) * 0.09
    - Drying temperature factor: (drying_temp - 23) * 0.15
    - Barrel open factor: barrel_open_hours * 0.5
    - Pre-fermentation factor: pre_fermentation_details * 0.1 if pre-fermentation == "Yes" else 0
    - Random noise: np.random.normal(0, 1.0)
    - Clip the final score within the range of 80 to 90.

## Conclusion

This document outlines the systematic approach for generating a synthetic dataset for the Café Anima Matris project. By following these guidelines, users can ensure that the generated data is realistic, consistent, and suitable for further analysis and model training.



