##### Libraries #####
import re
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, OneHotEncoder,
                                   OrdinalEncoder, PolynomialFeatures,
                                   PowerTransformer, StandardScaler)
from skopt import BayesSearchCV

##### Getting to Know Data #####
# import data
train = pd.read_csv('datasets/marketing_training.csv')
test = pd.read_csv('datasets/marketing_test.csv')

# data types of columns
def df_datatypes(df):
    df_desc = pd.DataFrame(df.dtypes.value_counts().reset_index())
    df_desc.columns = ['Data Type', 'Count']
    return df_desc.sort_values('Count', ascending=False)

df_datatypes(train)

# categorical - describe
def df_describe_categorical(df):
    return df.select_dtypes(include=['object']).describe()

df_describe_categorical(train)

# numerical - describe
def df_describe_numerical(df):
    return df.select_dtypes(include=np.number).describe()

df_describe_numerical(train)

# dataframe missing
def df_missing_info(df):
    pd.set_option('display.max_columns', df.shape[1])
    pd.set_option('display.max_rows', df.shape[1])
    descriptive_df = pd.DataFrame()
    descriptive_df['column'] = df.columns
    descriptive_df['data type'] = df.dtypes.tolist()
    descriptive_df['# missing'] = [df[col].isnull().sum() for col in df]
    descriptive_df['% missing'] = np.round(descriptive_df['# missing'] / df.shape[0], 4)
    return descriptive_df

df_missing_info(train)

# numerical - correlogram
def graph_correlogram(df):
    sns.set_theme(style="white") 
    # Compute the correlation matrix
    corr = df.corr(numeric_only=True)
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))
    # Draw the heatmap with the mask and correct aspect ratio
    graph = sns.heatmap(corr, mask=mask, cmap='RdYlGn', vmax=.3, center=0, square=True, annot=True, linewidths=.5, cbar_kws={"shrink": .5})
    return graph

graph_correlogram(train)

# correlation matrix with p-values
def df_correlation_matrix(df):
    numerical = df.select_dtypes(include=['int64', 'float']).columns.tolist()
    print(f'Pearson Correlation Matrix with P-Values')
    print(f'[Coef in Btm Tri / p-Values in Up Tri]')
    print(f'*** for <0.001, ** for <0.01, * for <0.05')
    print(f'-----------------------------------------')
    return df[numerical].rcorr(method='pearson').round(3)

df_correlation_matrix(train)

# graph of all numeric data types
def graph_numeric_histograms(df):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    numeric_columns = df.select_dtypes(include=np.number)
    for col in numeric_columns:
        sns.histplot(df[col], kde=True, color='black')
        plt.title(f'Histogram of {col.title()}')
        plt.show()

graph_numeric_histograms(train)

##### Mutual Information Statistic with Missing Data #####
# separate x & y data
X = train.drop(['responded'], axis=1)
y = LabelEncoder().fit_transform(train['responded'])

# numeric pipeline
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('knn_impute', KNNImputer()))
numeric_pipeline_steps.append(('min-max', MinMaxScaler(feature_range=(1, 2))))
numeric_pipeline_steps.append(('box-cox', PowerTransformer(method='box-cox')))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# categorical pipeline
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('si_impute', SimpleImputer(strategy='constant', fill_value='missing')))
categorical_pipeline_steps.append(('oe', OrdinalEncoder(dtype=np.int64)))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# create transformer
cat_columns = X.select_dtypes(include=['object']).columns.tolist()
num_columns = X.select_dtypes(include=[np.number]).columns.tolist()

transformer_steps = []
transformer_steps.append(('cat', categorical_pipeline, cat_columns))
transformer_steps.append(('num', numeric_pipeline, num_columns))
preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

# preprocessing transformer
preprocessing_transformer

# preprocess X data
pp_X = preprocessing_transformer.fit_transform(X)
pp_X_columns = pd.Series(preprocessing_transformer.get_feature_names_out()).str.replace('num__|cat__', '', regex=True)
pp_X_df = pd.DataFrame(pp_X, columns=pp_X_columns)

# get discrete feature indices
discrete_features_for_mi = [ind for ind, li in enumerate(pp_X_df.columns) if li in cat_columns]

# run mutual_info_classif
mi_scores = mutual_info_classif(pp_X_df, y, discrete_features=discrete_features_for_mi, random_state=2022)

# make df of mi
mi_scores_df = pd.DataFrame({'Feature': X.columns, 'MI Scores': mi_scores}).sort_values('MI Scores', ascending=False)

# bar plot
plt.figure(figsize=(12, 12))
sns.barplot(y='Feature', x='MI Scores', data=mi_scores_df, color='black')
plt.title('Mutual Information Statistic')
plt.show()

# dataframe of mi scores
mi_scores_df

##### Model Preparation #####
# split into train, test data
X_train, X_valid, y_train, y_valid = train_test_split(
    train.drop('responded', axis='columns'),
    train['responded'],
    stratify=train['responded'],
    test_size=0.30,
    random_state=2022
    )

# Numeric Feature Pipeline
numeric_pipeline_steps = []
numeric_pipeline_steps.append(('imputer', IterativeImputer()))
numeric_pipeline_steps.append(('scaler', StandardScaler()))
numeric_pipeline_steps.append(('poly', PolynomialFeatures(degree=2)))
numeric_pipeline = Pipeline(steps=numeric_pipeline_steps)

# Categorical Feature Pipeline
categorical_pipeline_steps = []
categorical_pipeline_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
categorical_pipeline_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
categorical_pipeline = Pipeline(steps=categorical_pipeline_steps)

# Preprocessing Transformer
transformer_steps = []
transformer_steps.append(('cat', categorical_pipeline, make_column_selector(dtype_exclude=np.number)))
transformer_steps.append(('num', numeric_pipeline, make_column_selector(dtype_include=np.number)))
preprocessing_transformer=ColumnTransformer(transformers=transformer_steps)

# Display Transformer
set_config(display='diagram')
preprocessing_transformer

##### Models #####
# Models
models = []
models.append(('LR', LogisticRegression(solver='saga', penalty='elasticnet', class_weight='balanced', l1_ratio=0.5, max_iter=100_000, random_state=2022)))
models.append(('DC', DummyClassifier(strategy='most_frequent')))
models.append(('RF', RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=2022)))
models.append(('KNN', KNeighborsClassifier(weights='distance')))

##### Fitting Models #####
from sklearn.model_selection import cross_val_score

# results dataframe
results = pd.DataFrame()

# scoring used
scoring = 'roc_auc'

# CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=2022)

for name, model in models:
    # Pipeline
    model_pipeline_steps = []
    model_pipeline_steps.append(('transformer', preprocessing_transformer))
    model_pipeline_steps.append(('model', model))
    pipeline = Pipeline(steps=model_pipeline_steps)
    # CV results
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    temp_df = pd.DataFrame({name: pd.Series(abs(cv_results))})
    results = pd.concat([results, temp_df], axis='columns')
    # Mean +/- std
    msg = f'{name}: {cv_results.mean().round(5)} +/- {cv_results.std().round(5)}'
    print(msg)

# Algorithm Comparison Boxplot
sns.pointplot(y='model', x=scoring,
            data=pd.melt(results, var_name='model', value_name=scoring), palette='colorblind')
plt.title('Spot Check Algorithm Boxplots')
plt.show()

# print df
results
pd.DataFrame({
    'mean':results.mean(),
    'std':results.std()
    }).sort_values('mean', ascending=True)

##### Tune Model #####
# Model 
tune_model = LogisticRegression(solver='saga', penalty='elasticnet', max_iter=10_000, random_state=2022, class_weight='balanced')

# Scoring
scoring = 'roc_auc'

# Pipeline Steps
model_pipeline_steps = []
model_pipeline_steps.append(('transformer', preprocessing_transformer))
model_pipeline_steps.append(('model', tune_model))
model_pipeline = Pipeline(steps=model_pipeline_steps)

# Param Grid
tune_model.get_params()
model_pipeline.get_params()
param_grid = {
    'model__C': np.linspace(0, 10, 100),
    'model__l1_ratio': np.linspace(0, 1, 100)
    }

# BayesSearchCV
bs_model = BayesSearchCV(
    estimator=model_pipeline,
    search_spaces=param_grid,
    scoring=scoring,
	n_iter=30,
    n_jobs=-1,
    random_state=2022)

# Display Model
bs_model

# Fit Model on Train
bs_model.fit(X_train, y_train)

# best score
print(f'best {scoring} score: {abs(bs_model.best_score_).round(5)}')

# best params
best_param_df = pd.DataFrame(bs_model.best_params_.items(), columns=['Parameter', 'Value'])
best_param_df['Parameter'] = best_param_df['Parameter'].str.replace('model__', '')
best_param_df['Value'] = np.round(best_param_df['Value'], 4)
best_param_df

# best estimator
best_model_pipeline = bs_model.best_estimator_

# predict on Validation
preds = best_model_pipeline.predict(X_valid)

###### ROC-AUC #####
# ROC AUC on Validation Set
RocCurveDisplay.from_estimator(
    estimator=best_model_pipeline,
    X=X_valid,
    y=y_valid
    )
plt.show()

##### Confusion Matrix #####
# Confusion on Validation Set
ConfusionMatrixDisplay.from_estimator(
    estimator=best_model_pipeline,
    X=X_valid,
    y=y_valid,
    cmap='Blues',
    colorbar=False
    )
plt.show()

##### Predict on Test Set #####
# predict on test
test['responded'] = best_model_pipeline.predict(test)

# save to csv
test.to_csv('test_with_preds.csv', index=False)  

# Save Model
model_name = 'Logistic Regression'
model_file_name = f'models/{model_name} on {datetime.now().strftime("%Y %b %d at %H.%M.%S")}.pkl'
joblib.dump(best_model_pipeline, model_file_name)