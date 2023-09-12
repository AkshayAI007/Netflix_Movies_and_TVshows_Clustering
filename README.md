# Netflix_Movies_and_TVshows_Clustering!

![Screenshot 2023-09-12 211950](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/0da64563-2f2c-41f8-b4ed-0618d070be5f)

# Project Summary :
# Problem Statement :
This dataset consists of tv shows and movies available on Netflix as of 2019. The dataset is collected from Flixable which is a third-party Netflix search engine.
In 2018, they released an interesting report which shows that the number of TV shows on Netflix has nearly tripled since 2010. The streaming service’s number of movies has decreased by more than 2,000 titles since 2010, while its number of TV shows has nearly tripled. It will be interesting to explore what all other insights can be obtained from the same dataset.
Integrating this dataset with other external datasets such as IMDB ratings, rotten tomatoes can also provide many interesting findings.

# About the Data :
We have the data of which contains details of customers like id , age, gender and also contains the details of the customers vehicle

* Dataset info
1.Number of records: 7787
2.Number of features: 12

Features information:
The dataset contains features like:
 * show_id : Unique ID for every Movie / Tv Show
 * type : A Movie or TV Show
 * title : Title of the Movie / Tv Shows
 * director : Director of the Movie
 * cast : Actors involved in the movie / show
 * country : Country where the movie / show was produced
 * date_added : Date it was added on Netflix
 * release_year : Actual Release year of the movie / show
 * rating : TV Rating of the movie / show
 * duration : Total Duration - in minutes or number of seasons
 * listed_in : Generes
 * description: The Summary description
# Project Work flow
 * Importing Libraries
 * Loading the dataset
 * Data Summary
 * Data Cleaning & Data Analysis
 * Feature selection
 * Implementing different clustering methods
 * Conclusion
 * Future Work

# Performance Metrics:
# 1) Silhouette score

Silhouette score is a popular evaluation metric for clustering algorithms. It measures how well each data point fits into its assigned cluster compared to other clusters. The score ranges from -1 to 1, with a higher score indicating better-defined clusters.
Silhouette score is a useful metric for a positive business impact because it can help identify the optimal number of clusters for a dataset. This, in turn, can help companies make data-driven decisions and allocate resources more efficiently based on the distinct patterns and characteristics of each cluster.

# 2) Calinski-Harabasz score

The Calinski-Harabasz score, also known as the variance ratio criterion, is a measure of the ratio between the within-cluster dispersion and the between-cluster dispersion. It is calculated by taking the ratio of the sum of squares between groups to the sum of squares within groups, multiplied by the ratio of the number of observations to the number of clusters minus one.
In other words, the Calinski-Harabasz score measures how well separated the clusters are in the data and how compact the clusters are internally. A higher score indicates that the clusters are well separated and compact, while a lower score indicates that the clusters are not well separated or are not compact.

# 3) Davies-Bouldin score
The Davies-Bouldin score is a measure of the average similarity between each cluster and its most similar cluster, compared to the average dissimilarity between each cluster and its least similar cluster. It is calculated by taking the sum of the ratios of the within-cluster scatter and the between-cluster distances, divided by the number of clusters.

In other words, the Davies-Bouldin score measures how well separated the clusters are in the data and how distinct they are from each other. A lower score indicates that the clusters are well separated and distinct, while a higher score indicates that the clusters are not well separated or are not distinct from each other.

# WordCloud for the movie/shows :
# KMeans Clustering:
![Kmeans](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/9a1eee1f-cf22-42f3-a499-e55dcbbcbd3e)

# Hierarchical clustering :
![download](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/ea14bc21-873f-4f3b-9a9d-91a1fe48206b)

# Model Performance:
# Kmeans Clustering:
![Screenshot 2023-09-12 213021](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/4a88ac74-de7e-41ea-acbb-d3efe9332618)
# Hierarchical Clustering :
![Screenshot 2023-09-12 213049](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/5e9d3f11-d536-465c-a10a-074635a45943)

# DBSCAN Clustering:
![Screenshot 2023-09-12 213104](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/4cb0c5b7-5caf-424c-9cc3-6204ba563acc)

# Final Model Performance:
![Screenshot 2023-09-12 213325](https://github.com/AkshayAI007/Netflix_Movies_and_TVshows_Clustering/assets/110448324/988cb5d5-888b-4540-bd24-9137ab0c443d)

#Final model Selection :

After evaluating multiple machine learning models including K-Means Clustering , Hierarchical Clustering - Agglomerative, DBSCAN Clustering and Recommender System, we selected K-Means Clustering as our final prediction model.

We chose K-Means Clustering because it performed well on our evaluation dataset in terms of accuracy and computational efficiency. The model was able to cluster similar movies and TV shows together based on their shared attributes, which allowed us to make better recommendations to our users. Additionally, K-Means Clustering was relatively easy to implement and maintain, which made it a practical choice for our project.

While Hierarchical Clustering - Agglomerative and DBSCAN Clustering also showed promising results, it was computationally expensive and required more processing power and time to execute. The Recommender System, on the other hand, was limited in its ability to cluster movies and TV shows based on shared attributes, and instead relied heavily on user behavior data to make recommendations.

# K- Means Ckustering Model gives better results for calinski_harabasz_score (higher than others) and Davies-Bouldin score(lower than others) than Hierarchical Clustering and DBSCAN Clustering also gives good silhouette score.
