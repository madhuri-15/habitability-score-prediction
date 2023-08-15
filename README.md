# Habitability Score of House Predictions
*(End-to-end Supervised Machine Learning Project using Python)*

### Problem Statement

<u>Your task is to predict the habitability score of the house using Machine Learning.</u>

**Habitability Score** is measure of how comfortable and suitable house is for living and it ranges from 0 to 100.
- 0  means the house is uninhabitable and 
- 100 means the house is perfect.

### Dataset
For this project, we will be using a dataset from `HackerEarth - 'Get a Room: ML Hackathon'`. You can download the dataset from [kaggle](https://www.kaggle.com/datasets/jaisinghchauhan/get-a-room-ml-hackathon-hackerearth?select=train.csv).

There are total 39,496 rows and 15 columns, including the `Habitability_score` which represent habitability score of the property and this column is our output variable.

**Data description**
- Property_ID	 : Represents a unique identification of a property
- Property_Type	: Represents the type of the property( Apartment, Bungalow, etc)  
- Property_Area	: Represents the area of the property in square feets 
- Number_of_Windows	: Represents the number of windows available in the property 
- Number_of_Doors : Represents the number of doors available in the property
- Furnishing : Represents the furnishing type ( Fully Furnished, Semi Furnished, or Unfurnished )
- Frequency_of_Powercuts : Represents the average number of power cuts per week
- Power_Backup : Represents the availability of power backup 
- Water_Supply : Represents the availability of water supply ( All time, Once in a day - Morning, Once in a day - Evening, and Once in two days)  
- Traffic_Density_Score : Represents the density of traffic on a scale of  1 to  10 
- Crime_Rate : Represents the crime rate in the neighborhood ( Well below average, Slightly below average, Slightly above average, and  Well above average ) 
- Dust_and_Noise	: Represents the quantity of dust and noise in the neighborhood ( High, Medium, Low ) 
- Air_Quality_Index : Represents the Air Quality Index of the neighborhood 
- Neighborhood_Review : Represents the average ratings given to the neighborhood by the people  
- Habitability_score	: Represents the habitability score of the property 

### Type of Machine Learning Problem

Within machine learning, there are two basic approaches: 
- **Supervised Learning**
- **Unsupervised Learning**

The main difference is one uses labeled data to help predict outcomes, while the other does not.

- `Supervised learning` is a machine learning approach that’s uses the labeled datasets which contains both input and output data values. These datasets are designed to train or `supervise` algorithms into classifying data or predicting outcomes accurately. Using labeled inputs and outputs, the model can measure its accuracy and learn over time.

- `Unsupervised learning` uses machine learning algorithms to analyze and cluster unlabeled data sets. These algorithms discover hidden patterns in data without the need for human intervention (hence, they are `unsupervised`).

Supervised learning can be separated into two types of problems: *classification* and *regression*.

- `Classification` problems use an algorithm to accurately assign test data into specific categories, such as labeling emails as spam or not spam.

- `Regression` is another type of supervised learning method that uses an algorithm to understand the relationship between dependent and independent variables. Regression models are helpful for predicting numerical values based on different data points, such as sales revenue projections for a given business.

> This is **Supervised Machine Learning problem** as we have training data with input and output pair values. And our output variable is conitinuous data type meaning it can be any value between 0 and 100. Therefore, it is a **Regression** Supervised Learning Problem.

### WorkFlow

-  Data Collection - Download the data
-  EDA - Perform statistical analysis to understand the feature distribution & relationships.
-  Data Preparation - Prepare data for model training.
-  Model Building - Build a baseline model & other ML model.
-  Model Evaluation - Compare results and evaluate the models on Performance metrics.
-  Model Optimization - Optimize the best model using Cross-validation & Hyperparameter tunning to improve the accuracy.
-  Model Predictions - Make predictions on test data & perform error analysis.

### Performance Metric

`score = max(0, 100 * (metrics.r2_score(actual , predicted))`

### Libraries
We will use standard Machine Learning & Data Science libraries
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Notebook
- [Part - 1 : Exploratory Data Analysis](https://github.com/madhuri-15/habitability-score-prediction/blob/main/Part%20I%20-%20Exploratory%20Data%20Analysis.ipynb)
- [Part - 2 : Model Selection & Predictions](https://github.com/madhuri-15/habitability-score-prediction/blob/main/Part%20II%20-%20Model%20Selection%20%26%20Prediction.ipynb)
