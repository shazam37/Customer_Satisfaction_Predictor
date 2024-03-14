# Customer_Satisfaction_Predictor

Customer satisfation is the neck and break of any business. If any business fails to deliver on it, then it is doomed to fail. Thus, timely inspection of customers response plays a pivotal role in their working. Manual inspection takes a whole lot of time and that is why companies rely on Machine Learning Algorithms for this task. The customer satisafaction is rated between 1-5 which makes it a regression problem. 

Here I have obtained the Brazilian olist customer satisafaction data from the kaggle. It contains many features, but I chose to keep it simple and went along with a few features only. The dataset was cleaned appropriately and fed through a simple Linear Regression model. I have used ZenML for maintaining the CI/CD pipeline of my application. 

The project is still in work. I am planning to include time stamp features along with pincode data for training the future models. Also, I will incorporate other Regression Model for the task. The best R2 score obtained so far on vanilla linear regression (with l2 regularization) is only 0.7. 
