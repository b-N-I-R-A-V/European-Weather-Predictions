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
``` python
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

The confusion matrix for all the weather stations:
<div align ="center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/728ae7ff-3897-4022-b429-74ff916e6bd4">
</div>
<div align ="center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/22e419e5-303d-4cf1-9361-565c4ddafc17">
</div>



The following video provides a walkthrough of the techniques used and my opinion on selecting the best algorithm. 
[YouTube Video Presentation](https://youtu.be/WdBm0hqbXZY)
You can also go through the presentation file that I have created which can help in understanding the project better.

[Presentation File](https://github.com/b-N-I-R-A-V/European-Weather-Predictions/blob/main/Presentation%20File.pdf)


