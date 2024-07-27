# European Weather Prediction
**Purpose and Context:** This project was aimed at utilizing machine learning techniques to predict consequences and impact of extreme weather events. The project was part of my coursework of Machine Learning Specialization by CareerFoundry. In this project we have used several machine learning algorithms to predict pleasant and unpleasant days for outdoor activities and we have assesed which algorithm best work for this goal.

### Project Goals:
* Finding new patterns in weather changes over the last 60 years.
* Identifying weather patterns outside the regional norm in Europe.
* Determining whether unusual weather patterns are increasing.
* Generating possibilities for future weather conditions over the next 25 to 50 years based on current trends.
* Determining the safest places for people to live in Europe within the next 25 to 50 years.

## Tools and Techniques
For this project Python was used. We used Sci-kit learn, Tensorflow, and Keras as our primary libraries for machine learning algorithms along with data manipulation library pandas and for numerical computation library numpy. In particular we are using following tools and techniques:
* K-Nearest Neighbors(KNN), Decision Trees, Random Forests, Convolutional Neural Network, Recurrent Neural Network (Long-Short Term Memory), Generative Adversarial Network(GAN).
* Scaling Data to improve model performance.
* Gradient Descent method for predicting temperature.

## Project Hypotheses
* Machine learning can accurately predict historical extreme temperatures using temperature data from stations across Europe.
* Machine learning models can predict the likelihood of extreme weather events based on historical data.
* Machine learning models can be used to predict whether weather conditions on a given day will be favorable or unfavorable for outdoor activities.

## Dataset
The dataset for this project comes from European Climate Assessment & Dataset (ECA&D). They have robost data quality and homogeneity procedures to ensure data reliability and data consistency. The data contained in the dataset is between the year 1960-2022. This data is provided by various participating instituions across Europe.  

* [Link to the dataset used](https://s3.amazonaws.com/coach-courses-us/public/courses/da-spec-ml/Scripts/A1/Dataset-weather-prediction-dataset-processed.csv)
* [Pleasant Weather Data](https://github.com/b-nirav/European-Weather-Predictions/blob/main/Original%20Data/Dataset-Answers-Weather_Prediction_Pleasant_Weather.csv)


### Challenges with the Dataset
For the most recent data, there could be error present as it might not have been through proper validation processes. Further, there might be concerns around instrument malfunction or calibration errors present in the data. But considering the fact that they resolve any error which could be present in the historical data and the quality measures in place, this is the best dataset that we can have.

### Data Preprocessing
The dataset used was scaled using StandardScalar() to make sure that features with extreme values do not create bias in the dataset. Moreover, there were 18 weather stations in total. But pleasant day data was not available for 3 of the weather stations and hence weather data from these stations were dropped. Further, once I created a subset of the original data by selecting a weather station or time period, I removed month and date data from the subsets to train the models.


## Gradient Descent Method
To fulfill the objectives stated, I started exploring gradient descent method to identify a relationship between days of year and temperature. Gradient Descent is a type of optiization algorith where the primary objective is to minimize the loss function. There were two parameters which approximated temperature, namely theta_0 and theta_1. Following code was used to calculate the loss function.
``` python
def compute_cost(X, y, theta=np.array([[0],[0]])):
    """Given covariate matrix X, the prediction results y and coefficients theta
    compute the loss"""
    
    m = len(y)
    J=0 # initialize loss to zero
    
    # reshape theta
    theta=theta.reshape(2,1)
    
    # calculate the hypothesis - y_hat
    h_x = np.dot(X,theta)
    #print(h_x)
    
    # subtract y from y_hat, square and sum
    error_term = sum((h_x - y)**2)
    
    # divide by twice the number of samples - standard practice.
    loss = error_term/(2*m)
    
    return loss
```
The below is illustration for Mean Temperature of Madrid in 2020.

<div align = "center">
 <img width="70%" alt="image" src="https://github.com/user-attachments/assets/e8b5c733-d991-4b8d-924e-5e8831bbff8a">
</div>

* After about 100 iterations the loss function does not reduce further and the value of theta_0 and theta_1 stabilizes.
* For any station, and any year we can use gradient descent method to present temperature in terms of day of year as demonstrated here.

## Searching for Algorithm for Weather Prediction

One of the primary task of the project was to use the dataset and classify the days as pleasant or unpleasant day. This same alogorithm can be extended and used for identifying extreme weather consditions.

### K-Nearest Neighbours(KNN)
I started with one of the more simplar method for classification. Of course, I used the scaled dataset to train the model. And, I used train_test_split method from sklearn.model_selection to split the dataset in the training and testing dataset.
```python
#
k_range = np.arange(1,5)
train_acc = np.empty(len(k_range))
test_acc = np.empty(len(k_range))
scores = {}
scores_list = []
for i, k in enumerate(k_range):
    print("i -", i)
    print("k -", k)
    knn = KNeighborsClassifier(n_neighbors=k)
    classifier = MultiOutputClassifier(knn, n_jobs=-1) 
    ## Fit the model on the training data.
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    train_acc[i] = knn.score(X_train, np.array(y_train)) 
    test_acc[i] = knn.score(X_test, np.array(y_test)) 
```

Following graph shows the training and testing accuracy for various number of neighbours:
<div align = "center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/8c6a19d1-9305-4769-83ed-824f9fa5313e">
</div>
We started with number of neighbors equal to 1 and we gradually increase it to 4 neighbors. From the graph above we see that train accuracy falls sharply from 1 neighbor to 2 neighbors. The accuracy is reduced from 100% to 56%. The accuracy remains the same when we change the neighbors to 3 and it falls slightly when we increase the neighbors to 4, about 52% accuracy. In contrast, the train accuracy rises slowly from just above 42% accuracy when 1 neighbor to 45% accuracy when there are 4 neighbors.

The confusion matrix for all the weather stations:
<div align ="center">
    <img width="90%" alt="image" src="https://github.com/user-attachments/assets/728ae7ff-3897-4022-b429-74ff916e6bd4">
</div>
<div align ="center">
    <img width="90%" alt="image" src="https://github.com/user-attachments/assets/22e419e5-303d-4cf1-9361-565c4ddafc17">
</div>

Following were my observations:
* Considering the result of the test accuracy, I would say the algorithm is doing an average job of predicting the output.
* From the confusion matrix, it seems that the accuracy is high for predicting unpleasant days.
* In comparison, it is doing a poorer job of predicting the pleasant days.
* The algorithm is giving 100% accuracy for Sonnblick station as there is only 1 output for any combination of input.

### Decision Trees
Decision Trees are useful machine learning algorithms for classification. Decision Trees work by recursively splitting the data into subsets based on input features. They try create homogeneous subsets by calculating impurity of a node and reducing the impurity to 0.

There were two important hyperparameters that I looked at, criterion and min_samples_split. I considered "gini"
criterion to split the nodes of the tree and checked performance for different values of min_samples_split.
```python
weather_dt = DecisionTreeClassifier(criterion='gini', min_samples_split=10)
weather_dt.fit(X_train, y_train)

#training accuracy score using the cross validation method
y_pred_train = weather_dt.predict(X_train)
print('Train accuracy score: ',cross_val_score(weather_dt, X_train, y_train, cv = 3, scoring='accuracy').mean())

# y_test predictions
y_pred_test = weather_dt.predict(X_test)
# Accuracy of test data
accuracy_score(y_test, y_pred_test)
```
Train accuracy score:  0.6263072553707182
Test accuracy score: 0.6573719065876612

A snapshot from all the confusion matrices:
<div align = "center">
 <img width="80%" alt="image" src="https://github.com/user-attachments/assets/f309fbe9-b95d-49b9-aff6-8962919c05ff">
</div>

I tried pruning the decision tree by increasing the ‘min_samples_split’ parameter. I found that that the overall accuracy for the testing data is improved a little but in some cases the accuracy has become worse. For instance, in the case of Valentia, the pleasant days accuracy was very poor. The model predicted it correctly only 70 times out of 132 predictions! It might be due to the case that there are more unpleasant days recorded than there are pleasant days recorded. Therefore, I don’t think pruning the decision tree might help.


### Multilayer Perceptron Model
One of the more basic but effective artifical neural network is Multilayer-Perceptron Model. This model takes 3 important hyperparameters, number of hidden layers and neurons, maximum iterations, and tolerance level. Following is one of the model that I developed to assess its' accuracy:

```python
#Creating an ANN with 3 hidden layers 25 nodes each and 500 iterations
mlp = MLPClassifier(hidden_layer_sizes=(25,25,25), max_iter=500, tol=0.0001)

#Fit the data to the model
mlp.fit(X_train, y_train)
```
Following are the results:
<div align = "center">
    <img width="85%" alt="image" src="https://github.com/user-attachments/assets/0c17cf8f-b800-42cd-9f15-d262a39ce9fb">

</div>

In my opinion, three hidden layers of 35 nodes each with 1000 iterations and 0.0001 tolerance level worked the best for testing accuracy. Though, we will use other neural networks with regularization techniques to improve these results.


### Hierarchical Clustering
I then looked at an unsupervised machine learning algorithms to find if they can produce some meaningful clusters. I plotted dendrograms to see how these clusters are formed. There were several methods to calculate the distance between clusters such as 'single', 'complete', 'average', and 'ward' method. I looked at all the methods for selected years and created a crosstab to find the intersection of clusters created and pleasant and unpleasant days.

```python
# Clusters and Dendograms using 'ward' method
dist_sin = linkage(df1_scaled,method="ward")
plt.figure(figsize=(18,6))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('All Stations, Year 2010')
plt.suptitle("Dendrogram Ward Method",fontsize=18)
plt.show()
```
<div align = "center">
  <img width="90%" alt="image" src="https://github.com/user-attachments/assets/5ea82653-7ffc-44c8-b9fd-524f146bffd8">
</div>

There are 2 major clusters with distance matric being on the higher side. The size of the clusters also seems to be of the same size as with average method.

The above example considered all weather stations across Europe for 2010. We can either focus on all weather stations across Europe at a time or look at individual weather stations at a time. We can also have control or how the between clusters is to be calculated.

The below demonstration shows crosstab for weather station DUSSELDORF.

```python
df1_AM[['DUSSELDORF_pleasant_weather','STOCKHOLM_pleasant_weather']] = 0
df1_AM.loc[:, ['DUSSELDORF_pleasant_weather','STOCKHOLM_pleasant_weather']] = ans.loc[ans['DATE'].dt.year == 2010, ['DUSSELDORF_pleasant_weather','STOCKHOLM_pleasant_weather']].values

#Cluster and pleasant days for DUSSELDORF
print('Dusseldorf pleasant days:\n')
pd.crosstab(index = [df1_AM['DUSSELDORF_pleasant_weather']],columns =df1_AM['cluster'])
```

```python
#Cluster and pleasant days for STOCKHOLM
print('Stockholm pleasant days:\n')
pd.crosstab(index = [df1_AM['STOCKHOLM_pleasant_weather']],columns =df1_AM['cluster'])
```
<div align = "center">
  <img width="50%" alt="image" src="https://github.com/user-attachments/assets/9ef8f4b0-4eb3-4582-94b9-0cb6bb10f396">
    &nbsp; &nbsp; &nbsp; &nbsp;
  <img width="50%" alt="image" src="https://github.com/user-attachments/assets/f98f6947-d257-41e1-a949-ee87cb652938">
</div>

For both the weather station Dusseldorf and Stockholm, cluster 3 had almost all pleasant days. This means any day falling outside cluster 3 is likely to be unpleasant.

If we are to find new weather patterns over years, we can perhaps create clusters for each year. Then, we can analyze each these groups to see if there are any significant changes that has taken place over time.


The following video provides a walkthrough of the techniques used and my opinion on selecting the best algorithm. 
[YouTube Video Presentation](https://youtu.be/WdBm0hqbXZY)
You can also go through the presentation file that I have created which can help in understanding the project better.

[Presentation File](https://github.com/b-N-I-R-A-V/European-Weather-Predictions/blob/main/Presentation%20File.pdf)


