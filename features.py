"""Подготовка признаков для варианта 2 (Home Credit Default Risk, application_train.csv).

Модуль вынесен из ноутбука, чтобы сохранённый пайплайн загружался в другом процессе
(лабораторная работа № 2). Все функции не обучаются на данных: одинаково применяются
к обучающим и новым заявкам, поэтому утечки статистик из теста нет.
"""
import numpy as np
import pandas as pd

ID_COL = 'SK_ID_CURR'
TARGET = 'TARGET'
DAYS_EMPLOYED_CODE = 365243  # служебный код «не работает / пенсионер», а не стаж

# категориальные столбцы исходного файла (тип object)
CAT_COLS = [
    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FONDKAPREMONT_MODE',
    'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',
]

NEW_FEATURES = [
    'CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO', 'CREDIT_TERM',
    'GOODS_CREDIT_RATIO', 'AGE_YEARS', 'EMPLOYED_TO_AGE',
    'EXT_SOURCE_MEAN', 'EXT_SOURCE_STD', 'INCOME_PER_PERSON', 'DOCS_COUNT',
]


def housing_duplicates(columns):
    """Числовые признаки о жилье в вариантах _MODE и _MEDI дублируют _AVG."""
    return [c for c in columns
            if c.endswith(('_MODE', '_MEDI')) and c not in CAT_COLS
            and c != 'TOTALAREA_MODE']


def clean(X):
    """Исправление ловушек варианта: идентификатор, код 365243, XNA, дубли о жилье."""
    X = X.copy()
    X = X.drop(columns=[c for c in (ID_COL, TARGET) if c in X.columns])
    for c in X.columns:
        if c in CAT_COLS:
            X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce')
    X['DAYS_EMPLOYED_ANOM'] = (X['DAYS_EMPLOYED'] == DAYS_EMPLOYED_CODE).astype(int)
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(DAYS_EMPLOYED_CODE, np.nan)
    X['CODE_GENDER'] = X['CODE_GENDER'].replace('XNA', np.nan)
    return X.drop(columns=housing_duplicates(X.columns))


def add_features(X):
    """Десять новых признаков; DAYS_BIRTH заменяется возрастом в годах."""
    X = X.copy()
    income = X['AMT_INCOME_TOTAL'].replace(0, np.nan)
    X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / income
    X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / income
    X['CREDIT_TERM'] = X['AMT_CREDIT'] / X['AMT_ANNUITY'].replace(0, np.nan)
    X['GOODS_CREDIT_RATIO'] = X['AMT_GOODS_PRICE'] / X['AMT_CREDIT'].replace(0, np.nan)
    X['AGE_YEARS'] = -X['DAYS_BIRTH'] / 365.25
    X['EMPLOYED_TO_AGE'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    ext = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']]
    X['EXT_SOURCE_MEAN'] = ext.mean(axis=1)
    X['EXT_SOURCE_STD'] = ext.std(axis=1)
    X['INCOME_PER_PERSON'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS'].replace(0, np.nan)
    X['DOCS_COUNT'] = X.filter(like='FLAG_DOCUMENT_').sum(axis=1)
    return X.drop(columns='DAYS_BIRTH')


def to_category(X):
    """Категории для LightGBM: тип category, пропуск остаётся пропуском."""
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype('category')
    return X


def prepare(X):
    """Полная подготовка: очистка → новые признаки → категории."""
    return to_category(add_features(clean(X)))


def drop_columns(X, cols=()):
    """Удаление признаков, признанных бесполезными по абляции."""
    return X.drop(columns=[c for c in cols if c in X.columns])


def to_catboost(X):
    """CatBoost не принимает NaN в категориях: заменяем строкой 'missing'."""
    X = X.copy()
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = X[c].astype(object).where(X[c].notna(), 'missing').astype(str)
    return X
