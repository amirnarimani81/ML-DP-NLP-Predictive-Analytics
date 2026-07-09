<h1> Predictive Maintenance for Industrial Equipment Using Machine Learning</h1>

<p>
An end-to-end machine learning predictive maintenance system designed to detect industrial
equipment failures and classify failure types using sensor data. The project combines
data analysis, feature engineering, machine learning pipelines, imbalance handling,
hyperparameter optimization, and failure prediction.
</p>


<hr>

<h2>1.  Project Overview</h2>

<p>
Unexpected equipment failures can cause production downtime, maintenance costs, safety risks,
and operational inefficiencies. This project develops a predictive maintenance solution that
uses machine sensor data to identify early failure signals and support proactive maintenance decisions.
</p>

<p>
The system performs two main tasks:
</p>

<ul>
<li><b>Binary Classification:</b> Predict whether a machine will fail or not.</li>
<li><b>Multiclass Classification:</b> Identify the specific failure type.</li>
</ul>


<hr>

<h2>2.  Engineering / Business Problem</h2>

<p>
Industrial companies rely on continuous equipment operation. Traditional maintenance strategies
such as reactive maintenance can lead to unexpected breakdowns and expensive repairs.
</p>

<p>
The main challenges are:
</p>

<ul>
<li>Detecting failures before machine breakdown.</li>
<li>Handling highly imbalanced failure data.</li>
<li>Understanding the relationship between sensor behavior and machine health.</li>
<li>Building reliable prediction models for operational decisions.</li>
</ul>


<hr>

<h2>3.  Objectives</h2>

<ul>

<li>
Perform exploratory data analysis to understand sensor behavior and failure patterns.
</li>

<li>
Clean and preprocess industrial sensor data.
</li>

<li>
Engineer meaningful features from temperature, speed, torque, and tool wear measurements.
</li>

<li>
Develop machine learning classification models:
<ul>
<li>Logistic Regression</li>
<li>Decision Tree</li>
<li>Random Forest</li>
<li>Gradient Boosting</li>
<li>XGBoost</li>
<li>Support Vector Machine</li>
</ul>
</li>

<li>
Handle class imbalance using SMOTE.
</li>

<li>
Optimize models using GridSearchCV and K-Fold Cross Validation.
</li>

<li>
Evaluate models using Accuracy, Precision, Recall, F1-score, ROC-AUC, and PR-AUC.
</li>

</ul>


<hr>

<h2>4.  Dataset Description</h2>

<table border="1">

<tr>
<th>Property</th>
<th>Description</th>
</tr>

<tr>
<td>Dataset</td>
<td>Machine Predictive Maintenance Classification Dataset</td>
</tr>

<tr>
<td>Samples</td>
<td>10,000 records</td>
</tr>

<tr>
<td>Features</td>
<td>10 variables including machine sensor measurements</td>
</tr>

<tr>
<td>Missing Values</td>
<td>No missing values</td>
</tr>

<tr>
<td>Duplicates</td>
<td>No duplicate records</td>
</tr>

<tr>
<td>Target 1</td>
<td>Failure or Not (Binary Classification)</td>
</tr>

<tr>
<td>Target 2</td>
<td>Failure Type (Multiclass Classification)</td>
</tr>

</table>


<h3>Sensor Features</h3>

<ul>
<li>Air Temperature [K]</li>
<li>Process Temperature [K]</li>
<li>Rotational Speed [rpm]</li>
<li>Torque [Nm]</li>
<li>Tool Wear [min]</li>
<li>Product Quality Type (L, M, H)</li>
</ul>


<h3>Failure Categories</h3>

<ul>
<li>No Failure</li>
<li>Power Failure</li>
<li>Tool Wear Failure</li>
<li>Overstrain Failure</li>
<li>Random Failure</li>
<li>Heat Dissipation Failure</li>
</ul>


<hr>

<h2>Data Preprocessing Pipeline</h2>
<ul>

<li>
Numerical features:
<ul>
<li>Missing value imputation</li>
<li>StandardScaler normalization</li>
</ul>
</li>


<li>
Categorical features:
<ul>
<li>Most frequent value imputation</li>
<li>One-Hot Encoding</li>
</ul>
</li>

</ul>


<h3>Feature Engineering</h3>

<ul>
<li>Process temperature categorization (Low, Medium, High)</li>
<li>Temperature difference calculation</li>
<li>Tool wear analysis</li>
<li>Sensor trend analysis</li>
</ul>


<pre>

# Let's assume X_train is a pandas DataFrame
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


#  Preprocess FOR MAKE ON X_TRAIN AND X_TEST
 # X_train_prep = preprocessor.fit_transform(X_train)
# X_test_prep  = preprocessor.transform(X_test)

</pre>



<h3>ML & Pipeline</h3>

<pre>
smot = SMOTE(random_state=42)
X_smt, y_smt = smot.fit_resample(X_train_p, y_train)

print("After SMOTE:")
print("X_smt shape:", X_smt.shape)
print("y_smt distribution:\n", pd.Series(y_smt).value_counts())

def evaluate_model(model, model_name, X_train_p, y_train, X_test_p, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test_p)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    cv_scores = cross_val_score(model, X_train_p, y_train, cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)

    print("Threshold:", threshold)
    print("\nModel:", model.__class__.__name__)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", conf_matrix)
    print("Cross-Validation Accuracy Mean:", cv_mean)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("PR-AUC:", pr_auc)
    print("-" * 50)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "conf_matrix": conf_matrix,
        "cv_accuracy": cv_mean,
        "pr_auc": pr_auc}


results = {}

param_grid_lr = {'C': [0.1, 1, 10], 'solver': ['lbfgs']}
grid_lr = GridSearchCV(LogisticRegression(), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_smt, y_smt)

best_lr = grid_lr.best_estimator_
lr_results = evaluate_model(best_lr, 'Logistic Regression', X_smt, y_smt, X_test_p, y_test)



param_grid_dt = {
    "max_depth": [None, 3, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "criterion": ["gini", "entropy"]}

grid_dt = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid_dt,
    cv=5,
    scoring="accuracy",
    n_jobs=-1)

grid_dt.fit(X_smt, y_smt)
best_dt = grid_dt.best_estimator_
print("Best DT Params:", grid_dt.best_params_)

# Evaluate DT
dt_results = evaluate_model(best_dt, "Decision Tree", X_smt, y_smt, X_test_p, y_test)
</pre>





<h3>EDA Visualizations</h3>

<ul>
<li>Sensor distribution analysis</li>
<li>Failure distribution analysis</li>
<li>Correlation heatmap</li>
<li>Product type vs failure analysis</li>
</ul>


<img src="Assets/2.png" width="500">
<img src="Assets/pic3" width="500">


<hr>

<h2>6.  Technologies Used</h2>


<ul>

<li><b>Programming:</b> Python</li>

<li><b>Data Processing:</b> Pandas, NumPy</li>

<li><b>Visualization:</b> Matplotlib, Seaborn</li>

<li><b>Machine Learning:</b>
Scikit-learn, XGBoost
</li>

<li><b>Imbalanced Learning:</b>
SMOTE (imbalanced-learn)
</li>

<li><b>Optimization:</b>
GridSearchCV, K-Fold Cross Validation
</li>

<li><b>Deployment Potential:</b>
Flask / Streamlit Dashboard
</li>

</ul>


<hr>

<h2>7.  Results</h2>


<h3>Models Developed</h3>

<table border="1">

<tr>
<th>Model</th>
<th>Purpose</th>
</tr>

<tr>
<td>Logistic Regression</td>
<td>Baseline failure prediction model</td>
</tr>

<tr>
<td>Decision Tree</td>
<td>Rule-based failure classification</td>
</tr>

<tr>
<td>Random Forest</td>
<td>Ensemble-based prediction</td>
</tr>

<tr>
<td>Gradient Boosting</td>
<td>Improved nonlinear pattern detection</td>
</tr>

<tr>
<td>XGBoost</td>
<td>High-performance gradient boosting model</td>
</tr>

<tr>
<td>SVM</td>
<td>Nonlinear classification using kernel methods</td>
</tr>

</table>


<h3>Evaluation Metrics</h3>

<ul>

<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1-score</li>
<li>ROC-AUC</li>
<li>PR-AUC</li>
<li>Confusion Matrix</li>

</ul>




<hr>

<h2>8. Key Insights</h2>


<ul>


<li>
<b>Temperature Relationship:</b>
The difference between air temperature and process temperature provides useful information
about machine operating conditions.
</li>


<li>
<b>Tool Wear Impact:</b>
Increasing tool wear contributes significantly to failure probability and can be used
as an early warning indicator.
</li>


<li>
<b>Failure Imbalance:</b>
Machine failures represent a small percentage of total observations, therefore
SMOTE improves minority-class learning.
</li>


<li>
<b>Sensor-Based Prediction:</b>
Temperature, torque, rotational speed, and tool wear are important indicators
for predicting equipment health.
</li>


<li>
<b>Preventive Maintenance Benefit:</b>
The predictive system can reduce unexpected downtime, improve maintenance scheduling,
and increase equipment reliability.
</li>


</ul>

<hr>

<h2> Conclusion</h2>

<p>
This project demonstrates a complete predictive maintenance pipeline using industrial sensor
data. By combining preprocessing, feature engineering, imbalance handling, and machine learning
models, the system can identify potential equipment failures and support proactive maintenance
strategies for industrial operations.
</p>
