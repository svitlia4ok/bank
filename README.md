Welcome to the Mid-term Project!

Main Objective

My primary task will be to build a model to predict whether a bank customer will subscribe to a term deposit. Such tasks are common in various companies and domains when we want to understand whether a customer will buy a product, use our service/website in the next month, or purchase a subscription.


Dataset and ML Task

To accomplish this task, I will use a dataset from the UCI Machine Learning Repository. The data concerns direct marketing campaigns (telephone calls) of a Portuguese banking institution. The ultimate goal of the classification is to predict whether the customer will subscribe to a term deposit (target variable y).

I will work with the file bank-additional-full.csv, which can be downloaded [here](https://drive.google.com/file/d/1pDr0hAOnu1JsEiJeBu_F2Jv0GPxKiujW/view?usp=drive_link).

My Tasks for This Project

I need to build a solution for this binary classification problem. To do this:

	1.	Conduct Exploratory Data Analysis and hypothesize the influence of individual features on the target variable y.
	2.	Describe which methods I consider appropriate to use and choose a metric for evaluating model quality (the metric should be justified).
	3.	Perform data preprocessing for further input into the model. Preprocessing should include:
	•	Handling categorical variables (if required by the model)
	•	Grouping categories in categorical variables if necessary
	•	Filling missing values if any
	•	Detecting outliers and making decisions on how to handle them
	•	Creating additional features that I believe will improve the quality of ML models.
	4.	Train four different types of machine learning models, which must include:
	•	Logistic Regression
	•	kNN
	•	Decision Tree
	•	At least one boosting algorithm
	5.	Create a table comparing model quality indicating:
	•	Model name
	•	Hyperparameters
	•	Quality metric on the training set
	•	Quality metric on the validation set
	•	Comments on the model - whether it is good or not, whether it should be used or not, and any further experimental ideas regarding this model.
	6.	For the boosting algorithm, perform hyperparameter tuning in two ways:
	•	Sklearn: Randomized Search
	•	Hyperopt: Bayesian Optimization. Identify optimal hyperparameters and draw conclusions about model quality in each case.
	7.	Display feature importance for the model that performed the best and describe whether I consider this prioritization of feature importance to be reasonable from a common sense perspective.
	8.	Analyze records where the model made errors and based on this analysis, indicate how I can improve the existing solution to avoid current mistakes.
