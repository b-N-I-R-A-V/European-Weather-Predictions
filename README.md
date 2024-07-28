# European Weather Prediction
**Purpose and Context:** This project aimed to utilize machine learning techniques to predict the consequences and impacts of extreme weather events. The project was part of my coursework for the Machine Learning Specialization by CareerFoundry. In this project, we used several machine learning algorithms to predict pleasant and unpleasant days for outdoor activities and assessed which algorithm works best for this goal.

### Project Goals:
* Finding new patterns in weather changes over the last 60 years.
* Identifying weather patterns outside the regional norm in Europe.
* Determining whether unusual weather patterns are increasing.
* Generating possibilities for future weather conditions over the next 25 to 50 years based on current trends.
* Determining the safest places for people to live in Europe within the next 25 to 50 years.

## Tools and Techniques
FFor this project, Python was used. We utilized Scikit-learn, TensorFlow, and Keras as our primary libraries for machine learning algorithms, along with the data manipulation library pandas and the numerical computation library NumPy. Specifically, we used the following tools and techniques:
* K-Nearest Neighbors (KNN), Decision Trees, Random Forests, Convolutional Neural Networks, Recurrent Neural Networks (Long Short-Term Memory), and Generative Adversarial Networks (GAN).
* Scaling data to improve model performance.
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
For the most recent data, there could be errors present as it might not have been through proper validation processes. Additionally, there might be concerns around instrument malfunction or calibration errors in the data. However, considering the efforts to resolve any errors in the historical data and the quality measures in place, this is the best dataset that we can have.

### Data Preprocessing
The dataset was scaled using StandardScaler() to ensure that features with extreme values do not create bias in the dataset. Moreover, there were 18 weather stations in total, but pleasant day data was not available for 3 of the weather stations. Hence, weather data from these stations were dropped. Further, once I created a subset of the original data by selecting a weather station or time period, I removed month and date data from the subsets to train the models.


## Gradient Descent Method
To fulfill the stated objectives, I started exploring the gradient descent method to identify a relationship between the day of the year and temperature. Gradient Descent is a type of optimization algorithm where the primary objective is to minimize the loss function. There were two parameters which approximated temperature, namely theta_0 and theta_1. The following code was used to calculate the loss function.
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

One of the primary tasks of the project was to use the dataset and classify the days as pleasant or unpleasant. This same algorithm can be extended and used for identifying extreme weather conditions.

### K-Nearest Neighbours(KNN)
I started with one of the simpler methods for classification. Of course, I used the scaled dataset to train the model. Additionally, I used the train_test_split method from sklearn.model_selection to split the dataset into training and testing datasets.
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
I started with the number of neighbors equal to 1 and gradually increased it to 4 neighbors. From the graph above, we see that test accuracy falls sharply from 1 neighbor to 2 neighbors, reducing from 100% to 56%. The accuracy remains the same when we change to 3 neighbors and falls slightly when we increase to 4 neighbors, about 52% accuracy. In contrast, the train accuracy rises slowly from just above 42% with 1 neighbor to 45% with 4 neighbors.

The confusion matrix for all the weather stations:
<div align ="center">
    <img width="90%" alt="image" src="https://github.com/user-attachments/assets/728ae7ff-3897-4022-b429-74ff916e6bd4">
</div>
<div align ="center">
    <img width="90%" alt="image" src="https://github.com/user-attachments/assets/22e419e5-303d-4cf1-9361-565c4ddafc17">
</div>

Following were my observations:
* Considering the test accuracy results, I would say the algorithm is doing an average job of predicting the output.
* From the confusion matrix, it seems that the accuracy is high for predicting unpleasant days.
* In comparison, it is doing a poorer job of predicting pleasant days.
* The algorithm is giving 100% accuracy for the Sonnblick station as there is only 1 output for any combination of input.


### Decision Trees
Decision Trees are useful machine learning algorithms for classification. Decision Trees work by recursively splitting the data into subsets based on input features. They try to create homogeneous subsets by calculating the impurity of a node and reducing the impurity to 0.

There were two important hyperparameters that I looked at, criterion and min_samples_split. I considered the "gini" criterion to split the nodes of the tree and checked performance for different values of min_samples_split.
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

I tried pruning the decision tree by increasing the min_samples_split parameter. I found that the overall accuracy for the testing data improved a little, but in some cases, the accuracy worsened. For instance, in the case of Valentia, the pleasant days accuracy was very poor. The model predicted it correctly only 70 times out of 132 predictions! This might be because there are more unpleasant days recorded than pleasant days. Therefore, I don’t think pruning the decision tree might help.

### Multilayer Perceptron Model
One of the more basic but effective artificial neural networks is the Multilayer Perceptron Model. This model takes 3 important hyperparameters: the number of hidden layers and neurons, maximum iterations, and tolerance level. Following is one of the models that I developed to assess its accuracy:

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

In my opinion, three hidden layers of 35 nodes each, with 1000 iterations and a 0.0001 tolerance level, worked the best for testing accuracy. However, we will use other neural networks with regularization techniques to improve these results.


### Hierarchical Clustering
I then looked at unsupervised machine learning algorithms to see if they could produce meaningful clusters. I plotted dendrograms to observe how these clusters are formed. There were several methods to calculate the distance between clusters, such as the 'single', 'complete', 'average', and 'ward' methods. I examined all the methods for selected years and created a crosstab to find the intersection of clusters created and pleasant and unpleasant days.

```python
# Clusters and Dendograms using 'average' method
dist_sin = linkage(df1_scaled,method="average")
plt.figure(figsize=(18,6))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('All Stations, Year 2010')
plt.suptitle("Dendrogram Average Method",fontsize=18)
plt.show()
```
<div align = "center">
  <img width="90%" alt="image" src="https://github.com/user-attachments/assets/5ea82653-7ffc-44c8-b9fd-524f146bffd8">
</div>

There are two major clusters immediately visible. There are also two small clusters, with one cluster having only one point.

The above example considered all weather stations across Europe for 2010. We can either focus on all weather stations across Europe at once or look at individual weather stations separately. We can also control how the distance between clusters is calculated.

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
  <img width="45%" alt="image" src="https://github.com/user-attachments/assets/a729dc36-7d49-4c4f-9612-b55b31065e6e">
    &nbsp; &nbsp; &nbsp; &nbsp;
  <img width="45%" alt="image" src="https://github.com/user-attachments/assets/f98f6947-d257-41e1-a949-ee87cb652938">
</div>

For both the weather stations Dusseldorf and Stockholm, cluster 3 had almost all pleasant days. This means any day falling outside cluster 3 is likely to be unpleasant.

I think if we aim to find new weather patterns over the years, we can create clusters for each year. Then, we can analyze how these groups change to see if there are any significant changes that have taken place over time.


### Random Forests
Random forests are an ensemble learning algorithm that combines multiple decision trees to improve predictive performance. We can use them to identify important features related to extreme weather events. I used them for classification purposes using weather data from all stations for the period between 2012 and 2022. I used GridSearch() to find optimal hyperparameters to build this model.


```python
# creating a RF classifier
clf = RandomForestClassifier()

# Defining Grid Space
grid_space={'max_depth':[3,5,None],
              'n_estimators':[100,200],
              'max_features':[5,10,None],
              'min_samples_leaf':[2,3],
              'min_samples_split':[2,3,5],
             'criterion':['gini','entropy']
           }

start = time.time()
grid = GridSearchCV(clf,param_grid=grid_space,cv=3,scoring='accuracy', verbose=3, n_jobs=-1)
model_grid = grid.fit(X_train, y_train)
print('Search took %s minutes' % ((time.time() - start)/60))

# grid search results
print('Best GRID search hyperparameters are: '+str(model_grid.best_params_))
print('Best GRID search score is: '+str(model_grid.best_score_))
```
Best GRID search hyperparameters are: {'criterion': 'entropy', 'max_depth': None, 'max_features': None, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
Best GRID search score is: 0.6546885694729637

```python
# performing predictions on the test dataset
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("Model Accuracy: ", metrics.accuracy_score(y_test, y_pred))
```
Random Forest Classifier (with optimization) 61%

<div align = "center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/3f49116c-8954-48ad-b1f7-6e39f3ab722e">
</div>

The decision tree from the random forests classifier is very complex and incomprehensible. When I looked at individual stations with all years of data and performed GridSearch(), it was much simpler and comprehensible. Below is a model built for Maastricht for all years of data.
```python
#Grid_Space
grid_space={'max_depth':[2,3,5,None],
              'n_estimators':[50,100],
              'max_features':[5,10],
              'min_samples_leaf':[1,2,3],
              'min_samples_split':[2,3,5],
             'criterion':['gini','entropy']
           }
#Grid Search
grid = GridSearchCV(clf,param_grid=grid_space,cv=3,scoring='accuracy', verbose=3, n_jobs=-1)
model_grid = grid.fit(X_train, y_train)

# grid search results
print('Best GRID search hyperparameters are: '+str(model_grid.best_params_))
print('Best GRID search score is: '+str(model_grid.best_score_))
```
Best GRID search hyperparameters are: {'criterion': 'gini', 'max_depth': 3, 'max_features': 10, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
Best GRID search score is: 1.0

```python
# performing predictions on the test dataset
clf1 = RandomForestClassifier(n_estimators = 50, max_depth=3, max_features = 10, min_samples_leaf = 1, min_samples_split = 2,
                             criterion = 'gini')  
clf1.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# using metrics module for accuracy calculation
print("Model Accuracy: ", metrics.accuracy_score(y_test, y_pred))
print("F1-Score: ", metrics.f1_score(y_test, y_pred))
```
Model Accuracy:  1.0
F1-Score:  1.0

The model had an accuracy of 100% and F-1 score of 100%. Following is one of the decison trees from the forest.
<div align = "center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/c27f99e6-da97-485d-8601-baedf6ea76de">
</div>

I then extracted important features for this decision tree.
```python
imp_features = clf1.feature_importances_
imp_features

%matplotlib inline
plt.style.use('fivethirtyeight')
# list of x locations for plotting
x_values = list(range(len(imp_features)))

plt.bar(x_values, imp_features, orientation = 'vertical')
plt.xticks(x_values, X_train.columns.to_list(), rotation='vertical',fontsize=10)
plt.ylabel('Importance',fontsize=10); plt.xlabel('Features',fontsize=10); plt.title('MAASTRICHT Feature Importance',fontsize=14);
plt.yticks(fontsize=10);
```
<div align = "center">
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/05d861b4-52fc-420d-b9c6-30cc1f1ab2f7">
</div>

In this case, maximum temperature, precipitation, and sunshine were important. Random forests can produce good resutls for individual stations.

### Convolutional Neural Network
A Convolutional Neural Network (CNN) is a deep learning algorithm and much more complex than a multilayer perceptron model. This model took over an hour to optimize as I trained it on the entire dataset due to the high number of hyperparameters involved. This required using BayesianOptimization() to find the best values for hyperparameters.
```python
# Create function
def bay_area(neurons, activation, kernel, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate): 
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU,'relu']
    
    neurons = round(neurons)
    kernel = round(kernel)
    activation = activationL[round(activation)]
    optimizer = optimizerL[round(optimizer)]
    batch_size = round(batch_size)
    
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)
    
    def cnn_model():
        model = Sequential()
        model.add(Conv1D(neurons, kernel_size=kernel,activation=activation, input_shape=(timesteps, input_dim)))
        #model.add(Conv1D(32, kernel_size=1,activation='relu', input_shape=(timesteps, input_dim)))
        
        if normalization > 0.5:
            model.add(BatchNormalization())
            
        for i in range(layers1):
            model.add(Dense(neurons, activation=activation)) #(neurons, activation=activation))
            
        if dropout > 0.5:
            model.add(Dropout(dropout_rate, seed=123))
            
        for i in range(layers2):
            model.add(Dense(neurons, activation=activation))
            
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(n_classes, activation='softmax')) #sigmoid softmax
        # model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy']) #categorical_crossentropy
        model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']) #categorical_crossentropy
        return model
        
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=2, patience=20)
    nn = KerasClassifier(build_fn=cnn_model, epochs=epochs, batch_size=batch_size, verbose=2)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train, scoring=score_acc, cv=kfold, fit_params={'callbacks':[es]}).mean()
    return score
```
```python
optimum = nn_opt.max['params']
print(optimum)
print(round(optimum['optimizer']))
```
{'activation': 2.79884089544096, 'batch_size': 460.14665762139765, 'dropout': 0.7296061783380641, 'dropout_rate': 0.19126724140656393, 'epochs': 90.97701940610612, 'kernel': 1.9444298503238986, 'layers1': 1.2391884918766034, 'layers2': 2.42648957444599, 'learning_rate': 0.7631771981307285, 'neurons': 60.51494778125466, 'normalization': 0.770967179954561, 'optimizer': 3.456569174550735}
3

I used these values to train the model which when trained gave an **accuracy of 96.96%.** 

```python
## Building CNN Model
epochs = 91
batch_size = 460
#n_hidden = 32

timesteps = len(X_train[0])
input_dim = len(X_train[0][0])
n_classes = 15 # There are 15 classes.
layers1 = 1
layers2 = 2
activation = 'softsign'
kernel = 2
neurons = 61
normalization = 0.770967179954561
dropout = 0.7296061783380641
dropout_rate = 0.19126724140656393
optimizer = 'Adadelta'
learning_rate = 0.7631771981307285

model = Sequential()
model.add(Conv1D(neurons, kernel_size=kernel, activation=activation, input_shape=(timesteps, input_dim)))

if normalization > 0.5:
    model.add(BatchNormalization())
    
for i in range(layers1):
    model.add(Dense(neurons, activation=activation))
    
if dropout > 0.5:
    model.add(Dropout(dropout_rate, seed=123))
    
for i in range(layers2):
    model.add(Dense(neurons, activation=activation))
    
model.add(MaxPooling1D())

model.add(Flatten())

model.add(Dense(n_classes, activation='softmax')) #softmax sigmoid

model.compile(loss='categorical_crossentropy', optimizer=Adadelta(learning_rate), metrics=['accuracy'])
```
```python
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)
```
<div align = "center"
    <img width="80%" alt="image" src="https://github.com/user-attachments/assets/dcaf5c43-4696-438a-ac80-7a3cf6ead86c">
</div>

This is a much more useful model when we want to predict unpleasant and pleasant days for the entire dataset containing all 15 weather stations. In comparison, Random Forests had 100% accuracy for one station at a time and about 62% for all the stations.

**Note:**
* I have not optimized Recurrent Neural Networks (RNN) such as Long Short-Term Memory (LSTM), which works better with time-series data and can also be used if we aim to predict extreme weather events.
* CNN works well with image data and can be used for the classification of weather images into different categories describing weather events. I have added the script for weather image classification in the scripts folder. It had an accuracy of 93.33%.
* Scripts for Generative Adversarial Networks (GAN) have been left out of this presentation to make it more focused on classification algorithms. Nonetheless, they are available in the scripts folder if you want to check them out. They have the potential to visualize how future weather events can look if we train them on a sufficiently large dataset of images with extreme weather events.

## Conclusion and Recommendations

Extreme weather prediction is a very complex task, and there are numerous ways it can be done. One approach is using historical datasets of extreme weather events and training machine learning models with them. Based on the models I have developed, I created three thought experiments that would help predict extreme weather events in the future.

1. Clustering algorithms such as hierarchical clustering may help us find changes in weather patterns over time.
2. Classification algorithms such as Random Forests or CNN can be used to identify weather patterns that are outside regional norms and determine important features that may help us predict extreme weather events.
3. Generative adversarial networks (GANs) have the potential to generate images that will help us visualize future extreme weather events.

All three algorithms fulfill different objectives for European weather prediction and can be further developed:

* The algorithms with the highest accuracy after optimization were Random Forests (100%) and CNN (95.96%).
* We can further develop the second thought experiment that predicts future events using Random Forests (for identifying key features), CNN (for capturing intricate patterns), and LSTM for forecasting future events.
* Algorithms can be trained using image data on extreme weather events.

Among all the possibilities I have discussed, I would choose Random Forests to identify important features for extreme weather predictions and CNN to predict extreme weather events, as it had the highest accuracy for the entire dataset.


I hope you have found something useful. If you have any questions or suggestions for me, feel free to reach out via my [LinkedIn](https://www.linkedin.com/in/nirav-bariya/) profile or [email](mailto:nkb.bariya@gmail.com) me.  Have a great rest of the day!
