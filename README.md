# Prediction Using Supervised ML

## Project Overview
This project is part of the Graduate Rotational Internship Program at The Spark Foundation. It consists of two main tasks: 
1. **Prediction Using Supervised Machine Learning**
2. **Species Segmentation Using K-Means Clustering**

## Author
Oluwashina Dedenuola

### Libraries Used
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
%matplotlib inline
```
### Dataset
The dataset used in this task was provided by the Spark Foundation.
### Steps
1. **Data Loading**: The data is imported and displayed successfully.
2. **Data Analysis**: Descriptive statistics of the data are generated.
3. **Data Visualization**: A scatter plot is created to show the relationship between hours studied and scores obtained.
4. **Regression Model**: A Linear Regression model is created to predict scores based on hours studied.
5. **Predictions**: The model is used to make predictions for given hours of study.
6. **Visualization of Regression Line**: The regression line is plotted on the scatter plot to visualize the predictions.

### Results
The model shows a high R-squared value of approximately 0.953, indicating a strong relationship between study hours and scores.

## Task 2: Species Segmentation Using K-Means Clustering

### Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
```

### Dataset
The dataset used for this task is the Iris dataset, which contains measurements of various features of different Iris species.

### Steps
1. **Data Loading**: The data is loaded from a CSV file provided by the Spark Foundation.
2. **Data Mapping**: The species names are mapped to numerical values for clustering.
3. **Data Visualization**: Scatter plots are created to visualize the distribution of data points based on sepal length and width.
4. **Clustering**: K-Means clustering is performed, and the optimal number of clusters is determined using the Elbow method.

### Results
The clusters are visualized, providing insights into the distribution and segmentation of different Iris species.

## Conclusion
This project demonstrates the application of supervised and unsupervised machine learning techniques in predictive analysis and clustering. It provides a practical understanding of regression analysis and clustering methods.

## Future Work
Further improvements could include:
- Testing additional algorithms for better accuracy.
- Applying feature scaling techniques for better clustering results.

## Acknowledgments
- The Spark Foundation for the opportunity to participate in the Graduate Rotational Internship Program.



