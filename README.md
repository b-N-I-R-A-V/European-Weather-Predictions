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



## Searching for Best Algorithm
The following video provides a walkthrough of the techniques used and my opinion on selecting the best algorithm. 
[YouTube Video Presentation](https://youtu.be/WdBm0hqbXZY)
You can also go through the presentation file that I have created which can help in understanding the project better.

[Presentation File](https://github.com/b-N-I-R-A-V/European-Weather-Predictions/blob/main/Presentation%20File.pdf)


