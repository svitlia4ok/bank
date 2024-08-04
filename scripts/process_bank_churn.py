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
    for column in flag_columns:
        columnName = column+'_F'
        mapping = {'unknown': np.nan, 'yes': 1, 'no': 0}
        df[columnName] = df[column].map(mapping)
    return df

def preprocess_month(df: pd.DataFrame, column: str):
    ordinal = OrdinalEncoder(categories=[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']])
    columnName = column+'_L'
    ordinal.fit(df[[column]])
    df[columnName] = ordinal.transform(df[[column]])
    return {
        'ordinalEncoder' : ordinal,
        'df' : df
    }

def preprocess_day(df: pd.DataFrame, column: str):
    ordinal = OrdinalEncoder(categories=[['mon', 'tue', 'wed', 'thu', 'fri']])
    columnName = column+'_L'
    ordinal.fit(df[[column]])
    df[columnName] = ordinal.transform(df[[column]])
    return {
        'ordinalEncoder' : ordinal,
        'df' : df
    }

def preprocess_age(df: pd.DataFrame, column: str):
    columnName = column+'_L'
    bins = [0, 20, 30, 40 , 50, 60, 100]
    labels = ['l_20', '20-30', '30-40', '40-50', '50-60','g_60']
    df[columnName] = pd.cut(df[column], bins, labels=labels, right=False)

def calculate_outliers_range(df: pd.DataFrame, column: str) -> Dict[str, float]:
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
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14,6)

    pltname = f'Нормалізований розподіл значень за категорією: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    pltname = f'Кількість даних за категорією: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
      ax.bar_label(container)



def getInputsAndTargets(df: pd.DataFrame, target_col: str) -> Dict[str, pd.DataFrame]:
    """
    Виділяє вхідні дані та цільові значення з DataFrame.

    Args:
        df (pd.DataFrame): Вхідний DataFrame з даними.

    Returns:
        dict: Словник з ключами 'inputs' і 'targets', що містять відповідні DataFrame.
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
    Масштабує числові колонки вхідних даних.

    Args:
        scalerObj (BaseEstimator): Об'єкт скейлера.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.
        number_cols_to_scale (list): Список числових колонок для масштабування.

    Returns:
        dict: Словник з ключами 'inputs_train' і 'inputs_val', що містять масштабовані дані.
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
    Масштабує числові колонки вхідних даних.

    Args:
        scalerObj (BaseEstimator): Об'єкт скейлера.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.
        number_cols_to_imput (list): Список числових колонок для імпутації.

    Returns:
        dict: Словник з ключами 'inputs_train' і 'inputs_val', що містять імпутовані дані.
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
    Кодує категоріальні колонки вхідних даних.

    Args:
        encoderObj (BaseEstimator): Об'єкт енкодера.
        categorical_cols (list): Список категоріальних колонок для кодування.
        inputs_train (pd.DataFrame): Тренувальні вхідні дані.
        inputs_val (pd.DataFrame): Валідаційні вхідні дані.

    Returns:
        dict: Словник з ключами 'inputs_train', 'inputs_val' і 'categories_encoded_cols', що містять закодовані дані та назви закодованих колонок.
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
    Попередньо обробляє дані, включаючи масштабування числових колонок і кодування категоріальних колонок.

    Args:
        df (pd.DataFrame): Вхідний DataFrame з даними.
        scaleNumeric (bool): Вказує, чи потрібно масштабувати числові колонки.

    Returns:
        dict: Словник з попередньо обробленими даними та об'єктами скейлера і енкодера.
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

def getMetricsData(model, inputs_train, targets_train, inputs_val, targets_val, column):
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

        df: pd.DataFrame, 
        scalerObj: BaseEstimator, 
        encoderObj: BaseEstimator, 
        number_cols: List[str], 
        number_cols_to_scale: List[str],
        categorical_cols: List[str],
        scaleNumeric: bool = True
    ) -> pd.DataFrame:
    """
    Попередньо обробляє нові дані з використанням існуючих об'єктів скейлера і енкодера.

    Args:
        df (pd.DataFrame): Вхідний DataFrame з новими даними.
        scalerObj (BaseEstimator): Існуючий об'єкт скейлера.
        encoderObj (BaseEstimator): Існуючий об'єкт енкодера.
        number_cols (list): Список числових колонок.
        number_cols_to_scale (list): Список числових колонок для масштабування.
        categorical_cols (list): Список категоріальних колонок для кодування.
        scaleNumeric (bool): Вказує, чи потрібно масштабувати числові колонки.

    Returns:
        pd.DataFrame: Попередньо оброблений DataFrame з новими даними.
    """
    if scaleNumeric:
        df[number_cols_to_scale] = scalerObj.transform(df[number_cols_to_scale])
    
    categories_encoded_cols = encoderObj.get_feature_names_out().tolist()
    df[categories_encoded_cols] = encoderObj.transform(df[categorical_cols])
    
    df = df[number_cols + categories_encoded_cols]
    
    return df