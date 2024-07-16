# Recommendations Reports on Future Model Development

## Table of Contents
1. [Overview](#overview)
2. [Limitations](#limitations)
3. [Recommendations](#recommendations)
4. [Client-Specific Adjustments](#client-specific-adjustments)
5. [Model Details](#model-details)
6. [Conclusion](#conclusion)

## Overview

This report provides future developers with detailed insights, suggestions, and recommendations on how to effectively build and improve the predictive model for the washed fermentation process. Additionally, it offers guidance on integrating real data for enhanced accuracy and reliability.

## Limitations

**Data Collection and Quality:**

- **Data Accuracy:** Inconsistent data entry practices and potential inaccuracies in the data can lead to unreliable model predictions. However, developing a reliable model requires high data accuracy.
- **Limited Data Range:** The synthetic data may not capture the full range of variability present in real-world conditions, leading to a less robust model. Additionally, limited temporal and seasonal coverage can further reduce the model's ability to generalize.
- **Synthetic Data Limitations:** While synthetic data is useful for initial model training, it may not fully encapsulate the complexities of real-world conditions. This limitation can impact the model's ability to generalize effectively to actual data. Additionally, the dataset comprises only 5000 rows, with each fermentation type contributing approximately 2500 rows, which may not be sufficient to capture all relevant variations.
- **SCA Score Factors:** The SCA score is developed to be influenced by multiple factors in the synthetic data. While the model has a reasonable performance, real-world data can present different challenges and variability. Therefore, the model may need to be adapted and refined when applied to real-world scenarios to maintain its predictive accuracy and robustness.

**Feature Engineering:**
- **Missing Relevant Features:** The current dataset may lack critical features such as soil type, altitude, and detailed weather conditions, which significantly influence the fermentation process. Including these features can provide a more comprehensive understanding of the factors affecting coffee quality.
- **Interaction Terms:** The model may not adequately capture interactions between different features, limiting its predictive power. For instance, the combined effect of temperature and humidity on the fermentation process might not be fully represented.
- **Honey Fermentation Specifics:** The honey fermentation process involves specific features such as the duration the barrel is open and the sealing time, which are not applicable to the washed fermentation process. Refining these features can enhance the model's accuracy for honey fermentation.
- **Potential Omissions:** Given the complexity of fermentation processes, there may be additional factors that have been omitted. Continuous evaluation and inclusion of such factors are essential to improve the model's comprehensiveness and predictive power.

**Model Selection and Training:**
- **Model Overfitting:** The current model may overfit the synthetic data, resulting in poor generalization to real-world data. High model complexity without proper regularization can lead to overfitting and increased computational cost.
- **Simplistic Assumptions:** During model development, several simplistic assumptions were made, such as fixed temperature ranges (18.0°C to 28.0°C) and humidity levels (60% to 90%). These assumptions may not accurately reflect the diversity of real-world conditions, where temperature and humidity can vary more widely and interact in complex ways, potentially affecting the fermentation process and the resulting coffee quality.

**Hyperparameter Tuning:**
- **Limited Hyperparameter Space:** The current hyperparameter tuning may not explore a sufficiently large parameter space, potentially missing optimal configurations. Manual tuning efforts can be time-consuming and may not yield the best results compared to automated methods.

**Model Evaluation:**
- **Single Metric Dependence:** Relying primarily on metrics like MSE and R² for evaluation may not provide a comprehensive view of model performance. Limited use of robust validation strategies, such as cross-validation, may lead to an overestimation of model performance.

**Integration with Real Data:**
- **Normalization and Scaling:** Differences in data normalization and scaling between synthetic and real data can affect model predictions. Ensuring consistent data preprocessing is crucial.
- **Handling Missing Values:** Inadequate strategies for handling missing values in real data can reduce model reliability. Developing robust imputation methods or using algorithms that can inherently handle missing data is essential.
- **Continuous Learning:** Lack of continuous model retraining can lead to outdated models that do not capture new trends in the data. Regular updates and retraining with new data are necessary to maintain model accuracy.

## Recommendations

**Data Collection and Quality:**
Validate real data rigorously to ensure consistency and accuracy. Implement thorough data validation checks to identify and correct discrepancies or errors. Collect data over an extended period and across different seasons to capture more variability in the fermentation and drying processes. This will help in building a more robust and generalizable model. Enhance synthetic data by supplementing it with real data as soon as possible to improve model accuracy and robustness.

**Feature Engineering:**
Include additional relevant features such as soil type, altitude, and detailed weather conditions (e.g., wind speed, sunlight intensity) to provide a deeper understanding of the influences on coffee quality. Investigate potential interactions between features, such as the combined effect of temperature and humidity on the fermentation process. This can offer valuable insights and improve model performance.

**Model Selection and Training:**
Explore advanced machine learning models such as ensemble methods (e.g., Random Forest, Gradient Boosting), deep learning models (e.g., neural networks), and hybrid models that combine multiple algorithms for improved performance. Regularly assess the complexity of the model to avoid overfitting and utilize techniques like cross-validation and regularization to ensure the model generalizes well to unseen data. Revisit and refine model assumptions to better reflect real-world conditions. Adjust temperature and humidity ranges based on actual data observations.

**Hyperparameter Tuning:**
Implement automated hyperparameter tuning methods like Bayesian Optimization to efficiently explore the hyperparameter space and identify the best settings for the model. Continue using grid and random search methods for hyperparameter tuning, but consider larger and more diverse parameter grids to explore a wider range of possibilities.

**Model Evaluation:**
Use a variety of evaluation metrics beyond MSE and R², such as Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE) to gain a comprehensive understanding of model performance. Employ robust validation strategies such as k-fold cross-validation and out-of-sample testing to ensure the model's performance is consistently high across different subsets of the data.

**Integration with Real Data:**
Ensure consistent data preprocessing by normalizing and scaling real data similarly to the training data to maintain consistency in model predictions. Handle missing values in real data by developing robust imputation methods or using algorithms that can inherently handle missing data. Adopt a continuous learning approach where the model is periodically retrained with new data to keep it updated with the latest trends and patterns in the fermentation process.

## Client-Specific Adjustments

To address client-specific feedback:

**Adjust Drying Process Duration:**
Update the data generation and calculation rules to reflect that the drying process should last about 1 to 2 months instead of 3 to 5 days. This change should be reflected in the data collection template and the model's input parameters, ensuring that the duration is calculated in days.

**Modify Weather Conditions Recording:**
Alter the data structure to allow multiple weather condition entries per day. For example, if a 4-day fermentation process records two different weather conditions per day, ensure the model can handle up to 8 weather condition entries. Update the code to accommodate this flexibility and accurately reflect the clients' recording practices.

## Model Details

### Gradient Boosting Model

**Model Selection:**
Gradient Boosting was selected as the best performing model due to its ability to handle various types of data and capture complex patterns. The model's performance was evaluated using metrics like Mean Squared Error (MSE) and R² score, with Gradient Boosting showing the highest accuracy and lowest error among the models tested.

**Hyperparameter Tuning:**
The hyperparameters of the Gradient Boosting model were optimized using GridSearchCV. The optimal parameters were found to be:
- n_estimators: 200
- learning_rate: 0.05
- max_depth: 3
- subsample: 0.7
- min_samples_split: 2
- min_samples_leaf: 1

**Model Training:**
The model was trained on the preprocessed data, which included one-hot encoding of categorical variables and normalization of numerical features. The training set consisted of 80% of the data, with the remaining 20% used for testing.

**Cross-Validation:**
Cross-validation was performed using a 5-fold approach to ensure the model's robustness and reliability. The results showed a consistent performance across different folds, indicating that the model generalizes well to unseen data.

### Model Evaluation Metrics

**Mean Squared Error (MSE):**
MSE measures the average squared difference between the predicted and actual values. A lower MSE indicates better model performance.

**R² Score:**
The R² score represents the proportion of the variance in the dependent variable that is predictable from the independent variables. A higher R² score indicates a better fit of the model.

**Model Performance:**
The Gradient Boosting model achieved an MSE of approximately 0.368 and an R² score of around 0.824, demonstrating its effectiveness in predicting the SCA scores based on the input features.

### Predicting SCA Scores

**Usage:**
The trained Gradient Boosting model can be used to predict SCA scores for new batches of coffee beans. The input data should be preprocessed in the same way as the training data, ensuring consistency in feature scaling and encoding.

**Example:**
The model can predict SCA scores for new data points by loading the trained model and applying it to the preprocessed input data. The predictions are added to the input data frame, providing an easy way to interpret the results.

## Conclusion

By following these detailed recommendations, future developers can build more effective and accurate predictive models for the washed fermentation process. Integrating real data with robust data handling and model validation techniques will enhance the reliability and usefulness of the model. Continuous improvement and adaptation to new data will ensure the model remains relevant and valuable for optimizing the quality of coffee beans.
