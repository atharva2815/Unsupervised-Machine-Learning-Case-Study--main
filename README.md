# Predict City-Cycle Fuel Consumption

This repository contains a study on predicting city-cycle fuel consumption (mpg) for automobiles using unsupervised machine learning techniques, specifically hierarchical clustering and K-means clustering. The goal is to explore the impact of clustering on feature engineering and preprocessing for regression analysis tasks related to fuel consumption prediction.

## Problem Statement

The primary objective of this study is to develop an effective machine learning model for accurately predicting the city-cycle fuel consumption in miles per gallon (mpg) for automobiles. The project aims to leverage unsupervised learning techniques, such as hierarchical clustering and K-means clustering, to identify patterns and group similar instances together, thereby improving the accuracy of regression models trained on the clustered datasets.

The following key aspects are addressed:

1. Cluster the dataset using hierarchical clustering to create distinct datasets for regression model training.
2. Implement regression models to predict the 'mpg' attribute accurately for each cluster.
3. Handle the mix of discrete and continuous attributes through appropriate preprocessing techniques.
4. Evaluate the accuracy of the regression models and assess the impact of clustering on prediction performance.
5. Explore the correlation between variables and identify any significant relationships that can enhance predictive modeling.

## Dataset

The dataset used in this study comprises 398 instances of automobiles, each with 9 attributes, including:

- 3 multivalued discrete attributes: cylinders, model year, and origin
- 5 continuous attributes: displacement, horsepower, weight, acceleration, and miles per gallon (mpg)

## Methodology

### Data Preprocessing

1. Handling missing values by imputation with median values.
2. Detecting and addressing outliers using the interquartile range (IQR) method.
3. Encoding categorical variables for compatibility with clustering algorithms.

### Exploratory Data Analysis

1. Visualizing the distribution of variables using histograms.
2. Examining correlations between variables using a heatmap.
3. Identifying potential issues like multicollinearity.

### Hierarchical Clustering

1. Applying the average linkage method for hierarchical clustering.
2. Visualizing the clustering process using dendrograms.
3. Determining the optimal number of clusters based on domain knowledge and visual inspection.

### K-means Clustering

1. Employing the K-means algorithm for clustering.
2. Determining the optimal number of clusters using the elbow method and silhouette analysis.
3. Evaluating clustering performance using silhouette scores.

### Regression Modeling

1. Training linear regression models on the original dataset.
2. Training separate regression models on the clusters obtained from hierarchical and K-means clustering.

### Comparative Analysis

1. Comparing the performance of regression models trained on the original dataset and clustered datasets.
2. Assessing the impact of clustering on prediction accuracy.

## Results

The study found that K-means clustering explained the highest variation in the dataset, with a difference of only 1% compared to hierarchical clustering. However, the authors suggest that a larger dataset may be needed to gain more clarity, as the current dataset of used cars does not include potentially useful variables such as the number of previous owners, the gender of previous owners, and the reason or purpose for which the cars were being used.

## Conclusion

The study demonstrated the application of unsupervised learning techniques, specifically clustering, in preprocessing and feature engineering for regression analysis tasks related to fuel consumption prediction. While the results showed potential for improving prediction accuracy through clustering, the authors acknowledge the limitations of the dataset and suggest incorporating additional relevant features for better explainability and model performance.

## Usage

To reproduce the analysis or build upon this study, you can clone this repository and explore the code and data files.

## Dependencies

The following Python libraries are required to run the code:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
