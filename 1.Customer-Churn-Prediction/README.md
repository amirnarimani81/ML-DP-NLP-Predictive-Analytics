<!DOCTYPE html>
<html lang="en">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Customer Churn Prediction - README</title>
</head>

<body>

<h1> Customer Churn Prediction Using Machine Learning</h1>

<p>
End-to-end machine learning pipeline for predicting customer churn in the telecommunications industry.
The project includes data preprocessing, exploratory analysis, feature engineering, model training,
optimization, evaluation, and business insights.
</p>


<hr>

<h2> Problem Statement</h2>

<p>
Customer churn causes revenue loss and increases customer acquisition costs.
The objective is to build predictive models to identify high-risk customers and support
data-driven retention strategies.
</p>


<h2> Objectives</h2>

<ul>
<li>Analyze customer behavior using EDA</li>
<li>Create a reusable preprocessing pipeline</li>
<li>Train multiple machine learning classifiers</li>
<li>Optimize models using GridSearchCV</li>
<li>Compare performance using Accuracy, F1-score, ROC-AUC</li>
<li>Identify important churn drivers</li>
</ul>


<hr>

<h2> Dataset</h2>

<table border="1">

<tr>
<th>Property</th>
<th>Description</th>
</tr>

<tr>
<td>Industry</td>
<td>Telecommunications</td>
</tr>

<tr>
<td>Records</td>
<td>5,880 Customers</td>
</tr>

<tr>
<td>Features</td>
<td>21 demographic, service, billing variables</td>
</tr>

<tr>
<td>Target</td>
<td>Customer Churn (Yes/No)</td>
</tr>

<tr>
<td>Missing Values</td>
<td>None</td>
</tr>

</table>


<h3>Key Features</h3>

<ul>
<li>Customer demographics</li>
<li>Contract type and tenure</li>
<li>Monthly charges and billing information</li>
<li>Internet and support services</li>
<li>Payment methods</li>
</ul>


<hr>

<h2> Exploratory Data Analysis</h2>

<p>
EDA was performed to understand churn patterns and customer behavior.
</p>

<img src="assets/tenure_churn.png" width="300">

<img src="assets/contract_churn.png" width="300">

<img src="assets/correlation_heatmap.png" width="300">


<p>
Key findings:
</p>

<ul>
<li>Low-tenure customers have higher churn probability.</li>
<li>Month-to-month contracts show higher churn rates.</li>
<li>Customer charges and service usage influence churn behavior.</li>
</ul>


<hr>

<h2> Machine Learning Pipeline</h2>

<p>
A reusable preprocessing pipeline was developed using ColumnTransformer.
</p>


<pre>
Raw Data
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Scaling + Encoding
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Evaluation
</pre>


<h3>Preprocessing</h3>

<pre>
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

# Pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))])

# Combine them
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)])

# Apply transformation
X_train_p = preprocessor.fit_transform(X_train)
X_test_p= preprocessor.transform(X_test)

</pre>

<hr>

<h2> Machine Learning Models</h2>

<table border="1">

<tr>
<th>Model</th>
<th>Description</th>
</tr>

<tr>
<td>Logistic Regression</td>
<td>Baseline interpretable classifier</td>
</tr>

<tr>
<td>Decision Tree + SMOTE</td>
<td>Handles class imbalance</td>
</tr>

<tr>
<td>Random Forest</td>
<td>Ensemble tree-based model</td>
</tr>

<tr>
<td>Gradient Boosting</td>
<td>Sequential boosting algorithm</td>
</tr>

<tr>
<td>XGBoost</td>
<td>Regularized high-performance boosting</td>
</tr>

<tr>
<td>CatBoost</td>
<td>Gradient boosting optimized for categorical data</td>
</tr>

<tr>
<td>KNN, Naive Bayes, SVM</td>
<td>Benchmark comparison models</td>
</tr>

</table>


<hr>

<h2> XGBoost Model</h2>

<p>
XGBoost provides high prediction accuracy through regularization and efficient tree learning.
</p>

<pre>
XGBClassifier(
n_estimators=100,
learning_rate=0.1,
max_depth=4,
reg_alpha=0.5,
reg_lambda=1.0)
</pre>


<img src="assets/cat_feature_importance.png" width="400">

<hr>


<h2> PCA/PIPELINE </h2>
<pre>

pipe_xgb = Pipeline([
    ("preprocess", preprocessor),   # preprocessing
    ("pca", PCA(n_components=10)),  # PCA with 10 components
    ("xgb", XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        reg_alpha=0.5,
        reg_lambda=1.0,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])


pipe_xgb.fit(X_train, y_train)


y_pred = pipe_xgb.predict(X_test)


accuracy_xgb = accuracy_score(y_test, y_pred)
print(f"Accuracy of XGBoost with PCA: {accuracy_xgb * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
</pre>



<h2> Decision Tree Model</h2>

<p>
Decision Tree was implemented for efficient gradient boosting and handling complex customer patterns.
</p>

<pre>
DT = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", DecisionTreeClassifier(
        criterion='gini',        # or 'entropy'
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
        splitter='best'          # choose ONE: 'best' or 'random'))])

DT.fit(X_train, y_train)
</pre>


<img src="assets/cat_feature_importance.png" width="400">


<hr>

<h2> Hyperparameter Optimization</h2>

<p>
GridSearchCV with cross-validation was applied to improve model performance.
</p>

<ul>
<li>10-fold cross-validation</li>
<li>ROC-AUC scoring</li>
<li>Parameter search optimization</li>
</ul>


<pre>
#  Build Pipeline
model = Pipeline([
    ("preprocess", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))])

#  Define Parameter Grid
param_grid = {
    "classifier__criterion": ["gini", "entropy"],
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_split": [2, 5, 10],
    "classifier__min_samples_leaf": [1, 2, 4],
    "classifier__max_features": [None, "sqrt", "log2"]}

# Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=10,
    scoring="roc_auc",     # best for churn
    n_jobs=-1,
    refit=True,
    return_train_score=True)

#  Train
grid.fit(X_train, y_train)

#  Results
print("Best Params:", grid.best_params_)
print("Best ROC-AUC:", grid.best_score_)
</pre>
<hr>

<h2> Model Comparison</h2>


<table border="1">

<tr>
<th>Model</th>
<th>Accuracy</th>
<th>F1 Score</th>
<th>ROC-AUC</th>
</tr>


<tr>
<td>Random Forest</td>
<td>0.520</td>
<td>0.480</td>
<td>0.56</td>
</tr>


<tr>
<td>XGBoost</td>
<td>0.542</td>
<td>0.510</td>
<td>0.59</td>
</tr>


<tr>
<td><b>CatBoost</b></td>
<td><b>0.548</b></td>
<td><b>0.520</b></td>
<td><b>0.60</b></td>
</tr>

</table>


<h2> Model Performance Comparison</h2>

<p>
All models were evaluated using Accuracy, F1-score, and ROC-AUC metrics.
</p>

<img src="assets/model_comparison_bars.png" width="450">




<pre>
# Store results
training_score = []
testing_score = []


# Define models
models = [
    MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42),
    RandomForestClassifier(n_estimators=100, random_state=42),
    GradientBoostingClassifier(random_state=42),
    LogisticRegression(max_iter=1000, random_state=42),
    DecisionTreeClassifier(random_state=42)
]

model_names = [
    "MLP Classifier",
    "Random Forest",
    "Gradient Boosting",
    "Logistic Regression",
    "Decision Tree"
]

# Evaluation function
def model_prediction(model, name):

    #model= Pipeline([
        #('preprocess', preprocessor),
        #('model', model)])
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred) * 100
    test_acc = accuracy_score(y_test, y_test_pred) * 100
    
    training_score.append(train_acc)
    testing_score.append(test_acc)
    
    print(f"Accuracy of {name} on Training Data: {train_acc:.2f}%")
    print(f"Accuracy of {name} on Testing Data: {test_acc:.2f}%\n")

</pre>


<h2>ROC Curve Comparison</h2>

<p>
ROC curves were generated to compare model discrimination performance across thresholds.
</p>

<img src="assets/model_comparison_roc.png" width="450">
<hr>



<hr>

<h2> Business Insights</h2>

<ul>

<li>
<b>Tenure:</b>
New customers have higher churn risk.
</li>

<li>
<b>Contracts:</b>
Month-to-month contracts are associated with higher churn.
</li>

<li>
<b>Services:</b>
Lack of technical support increases churn probability.
</li>

<li>
<b>Main Predictors:</b>
Tenure, Contract Type, and Monthly Charges are the strongest features.
</li>

</ul>

<h2> Repository Structure</h2>

<pre>

Customer-Churn-Prediction/

├── data/
├── notebooks/
├── assets/
├── models/
├── src/
├── requirements.txt
└── README.md

</pre>


<hr>

<h2> Technologies</h2>

<ul>

<li>Python</li>
<li>Pandas, NumPy</li>
<li>Scikit-learn</li>
<li>XGBoost</li>
<li>CatBoost</li>
<li>Matplotlib, Seaborn</li>
<li>imbalanced-learn (SMOTE)</li>
<li>GridSearchCV</li>

</ul>


<hr>

<h2> Future Improvements</h2>

<ul>
<li>SHAP/LIME explainability</li>
<li>Deep learning churn models</li>
<li>Streamlit deployment dashboard</li>
<li>Real-time churn prediction API</li>
<li>Customer segmentation integration</li>
</ul>


<hr>

<h2> Conclusion</h2>

<p>
This project demonstrates a complete customer churn prediction workflow.
CatBoost achieved the best performance with <b>54.8% accuracy</b> and
<b>0.60 ROC-AUC</b>, providing actionable insights for customer retention
and business decision-making.
</p>


</body>

</html>