from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import auc, roc_auc_score, f1_score, roc_curve
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt

def starting_preprocessing(df: pd.DataFrame, flag_columns: List[str]):
    """
    Performs initial preprocessing on the input DataFrame by mapping specific categorical values to numerical values.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to be processed.
        flag_columns (List[str]): A list of column names in the DataFrame that need to be mapped.

    This function creates new columns in the DataFrame for each specified flag column, appending '_F' to the column name.
    The mapping converts the values:
        - 'unknown' to NaN
        - 'yes' to 1
        - 'no' to 0

    Returns:
        pd.DataFrame: The modified DataFrame with new columns added based on the mapping.
    """
    for column in flag_columns:
        columnName = column+'_F'
        mapping = {'unknown': np.nan, 'yes': 1, 'no': 0}
        df[columnName] = df[column].map(mapping)
    return df

def preprocess_month(df: pd.DataFrame, column: str):
    """
    Encodes the specified month column in the input DataFrame using ordinal encoding.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to be processed.
        column (str): The name of the column in the DataFrame that contains month data to be encoded.

    This function applies an OrdinalEncoder to convert the month names (e.g., 'jan', 'feb', ..., 'dec') into ordinal values,
    appending '_L' to the original column name for the new encoded column.

    Returns:
        Dict[str, Any]: A dictionary containing the fitted ordinal encoder under the key 'ordinalEncoder'
                         and the modified DataFrame under the key 'df'.
    """
    ordinal = OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']])
    columnName = column+'_L'
    ordinal.fit(df[[column]])
    df[columnName] = ordinal.transform(df[[column]])
    return {
        'ordinalEncoder' : ordinal,
        'df' : df
    }

def preprocess_day(df: pd.DataFrame, column: str):
    """
    Encodes the specified day column in the input DataFrame using ordinal encoding.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to be processed.
        column (str): The name of the column in the DataFrame that contains day data to be encoded.

    This function applies an OrdinalEncoder to convert the day names (e.g., 'mon', 'tue', 'wed', 'thu', 'fri') into ordinal values,
    appending '_L' to the original column name for the new encoded column.

    Returns:
        Dict[str, Any]: A dictionary containing the fitted ordinal encoder under the key 'ordinalEncoder'
                         and the modified DataFrame under the key 'df'.
    """
    ordinal = OrdinalEncoder(categories=[['mon', 'tue', 'wed', 'thu', 'fri']])
    columnName = column+'_L'
    ordinal.fit(df[[column]])
    df[columnName] = ordinal.transform(df[[column]])
    return {
        'ordinalEncoder' : ordinal,
        'df' : df
    }

def preprocess_age(df: pd.DataFrame, column: str):
    """
    Bins the specified age column in the input DataFrame into categorical ranges.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to be processed.
        column (str): The name of the column in the DataFrame that contains age data to be binned.

    This function creates a new column by binning the age values into specified ranges:
        - 'l_20' for ages less than 20
        - '20-30' for ages from 20 to 29
        - '30-40' for ages from 30 to 39
        - '40-50' for ages from 40 to 49
        - '50-60' for ages from 50 to 59
        - 'g_60' for ages 60 and above
    The new column name is created by appending '_L' to the original column name.

    Returns:
        None: The function modifies the DataFrame in place and does not return anything.
    """
    columnName = column+'_L'
    bins = [0, 20, 30, 40 , 50, 60, 100]
    labels = ['l_20', '20-30', '30-40', '40-50', '50-60','g_60']
    df[columnName] = pd.cut(df[column], bins, labels=labels, right=False)

def calculate_outliers_range(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """
    Calculates the minimum and maximum values for identifying outliers in the specified column of the DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame containing data to be analyzed.
        column (str): The name of the column in the DataFrame for which to calculate the outlier range.

    The function computes the first (Q1) and third (Q3) quartiles of the specified column, calculates the interquartile range (IQR),
    and determines the minimum and maximum thresholds for outliers, defined as 1.5 times the IQR below Q1 and above Q3.

    Returns:
        Dict[str, float]: A dictionary containing the calculated 'min_value' and 'max_value' for identifying outliers.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    min_value = (Q1 - 1.5 * IQR)
    max_value = (Q3 + 1.5 * IQR)
    return {
        'min_value': min_value,
        'max_value': max_value
    }


def bi_cat_countplot(df, column, hue_column):
    """
    Creates two bar plots for visualizing the distribution and counts of a categorical column in the DataFrame,
    stratified by a second categorical column (hue).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be visualized.
        column (str): The name of the categorical column for which to plot the distribution and counts.
        hue_column (str): The name of the categorical column used to stratify the data in the plots.

    This function generates:
        1. A normalized distribution bar plot showing the percentage of each category in the specified column,
           broken down by the hue column.
        2. A count bar plot showing the absolute count of each category in the specified column,
           also broken down by the hue column.

    Each bar plot is annotated with the percentage or count values for clarity. The function arranges the plots
    side by side for comparison.

    Returns:
        None: The function displays the plots and does not return any value.
    """
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14,6)

    pltname = f'Normalized distribution of values by category: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[0], title=pltname)

    # Annotation of values in the bar plot
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    pltname = f'Count of data by category: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
      ax.bar_label(container)



def getInputsAndTargets(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Extracts input data and target values from a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with data.

    Returns:
        dict: A dictionary with keys 'inputs' and 'targets', containing the corresponding DataFrames.
    """
    input_cols = df.drop(columns=[target_col]).columns
    inputs = df[input_cols]
    targets = df[[target_col]]
    return {
        'inputs': inputs,
        'targets': targets
    }

def scaleInputs(scalerObj: BaseEstimator, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_scale: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Scales the specified numerical columns in the training and validation datasets.

    Args:
        scalerObj (BaseEstimator): An instance of a scaler object that implements the fit and transform methods.
        inputs_train (pd.DataFrame): The training dataset containing input features to be scaled.
        inputs_val (pd.DataFrame): The validation dataset containing input features to be scaled.
        number_cols_to_scale (List[str]): A list of column names in the training and validation datasets that need to be scaled.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the scaled training and validation datasets, with keys 'inputs_train' and 'inputs_val'.
    """
    scalerObj.fit(inputs_train[number_cols_to_scale])
    inputs_train[number_cols_to_scale] = scalerObj.transform(inputs_train[number_cols_to_scale])
    inputs_val[number_cols_to_scale] = scalerObj.transform(inputs_val[number_cols_to_scale])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val
    }

def imputInputs(imputerObj: BaseEstimator, inputs_train: pd.DataFrame, inputs_val: pd.DataFrame, number_cols_to_imput: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Imputes missing values in the specified numerical columns of the training and validation datasets.

    Args:
        imputerObj (BaseEstimator): An instance of an imputer object that implements the fit and transform methods.
        inputs_train (pd.DataFrame): The training dataset containing input features.
        inputs_val (pd.DataFrame): The validation dataset containing input features.
        number_cols_to_imput (List[str]): A list of column names in the training and validation datasets that require imputation.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary containing the imputed training and validation datasets, with keys 'inputs_train' and 'inputs_val'.
    """
    imputerObj.fit(inputs_train[number_cols_to_imput])
    inputs_train[number_cols_to_imput] = imputerObj.transform(inputs_train[number_cols_to_imput])
    inputs_val[number_cols_to_imput] = imputerObj.transform(inputs_val[number_cols_to_imput])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val
    }

def encodeInputs(encoderObj: BaseEstimator, categorical_cols: List[str], inputs_train: pd.DataFrame, inputs_val: pd.DataFrame) -> Dict[str, Any]:
    """
    Encodes categorical columns in the training and validation datasets using the specified encoder.

    Args:
        encoderObj (BaseEstimator): An instance of an encoder object that implements the fit and transform methods.
        categorical_cols (List[str]): A list of column names in the training and validation datasets that contain categorical data to be encoded.
        inputs_train (pd.DataFrame): The training dataset containing input features to be encoded.
        inputs_val (pd.DataFrame): The validation dataset containing input features to be encoded.

    Returns:
        Dict[str, Any]: A dictionary containing the encoded training and validation datasets, along with the list of newly created encoded column names,
                         with keys 'inputs_train', 'inputs_val', and 'categories_encoded_cols'.
    """
    encoderObj.fit(inputs_train[categorical_cols])
    categories_encoded_cols = encoderObj.get_feature_names_out().tolist()
    inputs_train[categories_encoded_cols] = encoderObj.transform(inputs_train[categorical_cols])
    inputs_val[categories_encoded_cols] = encoderObj.transform(inputs_val[categorical_cols])
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'categories_encoded_cols': categories_encoded_cols
    }

def preprocess_data(df: pd.DataFrame, number_cols_to_scale: List[str], categorical_cols: List[str], scaleNumeric: bool = True, targetColumn: str = 'Exited') -> Dict[str, Any]:
    """
    Preprocesses the data, including scaling numerical columns and encoding categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame with data..
        scaleNumeric (bool): Indicates whether to scale the numerical columns.

    Returns:
        dict: A dictionary with preprocessed data and the scaler and encoder objects.
    """
    it = getInputsAndTargets(df, targetColumn)
    inputs = it['inputs']
    targets = it['targets']
    inputs_train, inputs_val, targets_train, targets_val = train_test_split(inputs, targets, test_size=0.1, random_state=42, stratify=df[targetColumn])
    number_cols = inputs.select_dtypes(include="number").columns.to_list()
    
    scalerObj = MinMaxScaler()
    if scaleNumeric:
        scaled_data = scaleInputs(scalerObj, inputs_train, inputs_val, number_cols_to_scale)
        inputs_train = scaled_data['inputs_train']
        inputs_val = scaled_data['inputs_val']
    
    encoderObj = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded_data = encodeInputs(encoderObj, categorical_cols, inputs_train, inputs_val)
    inputs_train = encoded_data['inputs_train']
    inputs_val = encoded_data['inputs_val']
    categories_encoded_cols = encoded_data['categories_encoded_cols']

    imputerObj = SimpleImputer(strategy='median')
    imputed_data = imputInputs(imputerObj, inputs_train, inputs_val, number_cols_to_scale)
    inputs_train = imputed_data['inputs_train']
    inputs_val = imputed_data['inputs_val'] 
    
    inputs_train = inputs_train[number_cols + categories_encoded_cols]
    inputs_val = inputs_val[number_cols + categories_encoded_cols]
    
    return {
        'inputs_train': inputs_train,
        'inputs_val': inputs_val,
        'targets_train': targets_train,
        'targets_val': targets_val,
        'scalerObj': scalerObj,
        'encoderObj': encoderObj,
        'number_cols': number_cols,
        'categorical_cols': categorical_cols,
        'number_cols_to_scale': number_cols_to_scale
    }

def getMetricsData(model, inputs_train: pd.DataFrame, targets_train: pd.DataFrame, inputs_val: pd.DataFrame, targets_val: pd.DataFrame, column: str) -> Any:
    """
    Fits the specified model to the training data, evaluates its performance on both training and validation datasets,
    and prints various performance metrics.

    Args:
        model: The machine learning model to be trained and evaluated.
        inputs_train (pd.DataFrame): The training dataset containing input features.
        targets_train (pd.DataFrame): The training dataset containing target values.
        inputs_val (pd.DataFrame): The validation dataset containing input features.
        targets_val (pd.DataFrame): The validation dataset containing target values.
        column (str): The name of the target column in the datasets.

    This function performs the following steps:
        1. Fits the model using the training inputs and targets.
        2. Predicts the target values and their probabilities for the training dataset.
        3. Computes and prints the AUC, ROC AUC score, and F1 score for the training dataset.
        4. Predicts the target values and their probabilities for the validation dataset.
        5. Computes and prints the AUC, ROC AUC score, and F1 score for the validation dataset.

    Returns:
        The trained model after evaluation.
    """
    model.fit(inputs_train, targets_train[column])
    targets_train_pred = model.predict(inputs_train)
    targets_train_pred_proba= model.predict_proba(inputs_train)
    fpr, tpr, _ = roc_curve(targets_train[column], targets_train_pred_proba[:, 1])
    print("Train auc={}".format(auc(fpr, tpr)))
    print("Train roc_auc_score={}".format(roc_auc_score(targets_train[column], targets_train_pred)))
    print("Train f1={}".format(f1_score(targets_train[column], targets_train_pred)))

    targets_val_pred = model.predict(inputs_val)
    targets_val_pred_proba= model.predict_proba(inputs_val)
    fpr_v, tpr_v, _ = roc_curve(targets_val[column], targets_val_pred_proba[:, 1])
    print("Val auc={}".format(auc(fpr_v, tpr_v)))
    print("Val roc_auc_score={}".format(roc_auc_score(targets_val[column], targets_val_pred)))
    print("Val f1={}".format(f1_score(targets_val[column], targets_val_pred)))
    return model