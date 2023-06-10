# Hospitalization Prediction for Elderly Population

Hospitalizations can be a significant burden for the elderly, their families, and the healthcare system, and predicting the probabilities of hospitalization events can help in the early detection and prevention of adverse health outcomes.

This project aims to predict the likelihood of hospitalization events for Mexican adults over the age of 50 using data from the MHAS (Mexican Health and Aging Survey).

The survey provides information on various demographic, health, and lifestyle-related factors, including age, gender, marital status, education level, smoking habits, alcohol consumption, chronic conditions, physical limitations, and mental health status.

To achieve this, this project will involve building predictive models using supervised learning techniques. The models will be trained on a subset of the MHAS dataset, and the performance of each model will be evaluated based on its ability to predict hospitalization events accurately.

### Requirements:

The requirements for this project are the following ones:
- To perform an exploratory analysis of the dataset to understand its structure and characteristics.
- To develop scripts for data preprocessing and preparation.
- To Implement and train machine learning models using a subset of relevant features.
- Once the model is trained, it is required to achieve an AUC (Area Under the Curve) of at least 0.9 to assess the model's quality.
- Finally, to create an API implementing Docker, Flask and Redis.

### Scope & Main Deliverables:
1. We have a database with interviews from 2000 to 2018. In this database, more than 50,000 people were interviewed on various topics such as physical and mental state, social and economic status.
Firstly, we decided to focus on variables related to the individual’s health. We also included fields such as habits, received care and the interviewer’s assessment of their living conditions to improve the predictive power of determining whether they will be hospitalized or not.

2. One of the difficulties encountered in the project was dealing with "wide-format data" which presented several challenges. Firstly, the data contained numerous missing values, making it necessary to address the issue of handling missing data effectively.
Additionally, the dataset had a large number of columns, which posed challenges in terms of data exploration, analysis, and modeling. Working with such extensive datasets requires careful consideration of feature selection and dimensionality reduction techniques to avoid issues related to computational efficiency and model complexity.
Moreover, the project faced challenges related to underfitting and overfitting. Underfitting occurs when the model fails to capture the underlying patterns and relationships present in the data, resulting in poor performance and low predictive power. Overfitting, on the other hand, happens when the model becomes overly complex and starts to memorize the training data instead of learning general patterns. This can lead to poor performance on unseen data.
To address underfitting, techniques such as increasing the model complexity, using more sophisticated algorithms, or incorporating additional relevant features could be considered. Overfitting can be mitigated by regularization techniques, such as adding penalty terms or using ensemble methods.

The project plan was divided into the following milestones:
- Initial repository setup and project structure configuration.
- Review of the state of the art and understanding of the problem.
- Download and evaluation of the MHAS dataset.
- Creation of a clean and prepared training dataset.
- Training and evaluation of various machine learning models for hospitalization prediction.
- Presentation of the results and demonstration of the model in real-time using an API.
- Additional testing and preparation for the final presentation.

### Metrics:

In the current project we are working in a classification task, a binary one. In this case, it is suitable to use an accuracy  score. Along with these metrics we use the probability distribution of the value predicted in order to have a high level overview of the model performance. There are a couple of metrics we can focus on, namely the recall, and the precision. For this project the recall will be more appropriate because of the medical nature of the problem. Finally, we computed train and test data metrics in order to spot any over or underfitting issue.

## Project structure:

- In the `api` folder we can find the API configuration and form templates, is the user interface of the application.
- In the `model` folder we can find the configuration of the model, is the backend of the application.
- In the `model_training` folder we can find the development and analysis implemented for the used model, it also contains the project pipeline.