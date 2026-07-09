<h1> Credit Card Fraud Detection Using Machine Learning</h1>

<p>
An end-to-end machine learning project focused on detecting fraudulent credit card transactions
using imbalance learning techniques. The project investigates different resampling strategies
including SMOTE, Tomek Links, SMOTE-Tomek, SMOTEENN, and Random Oversampling to improve fraud detection performance.
</p>


<hr>

<h2>1.  Project Overview</h2>

<p>
Credit card fraud detection is a challenging classification problem because fraudulent
transactions represent a very small percentage of total transactions. Traditional models
can achieve high accuracy while failing to identify fraudulent cases.
</p>

<p>
This project develops a fraud detection pipeline that focuses on minority class detection
using appropriate evaluation metrics such as Precision, Recall, F1-score, and PR-AUC.
</p>


<hr>

<h2>2.  Engineering / Business Problem</h2>

<p>
Financial institutions process millions of transactions daily. Missing fraudulent transactions
can result in financial losses, customer dissatisfaction, and security risks.
</p>

<p>
The main engineering challenge is the extreme class imbalance:
</p>

<ul>
<li>Normal transactions dominate the dataset.</li>
<li>Fraud cases represent only a very small fraction.</li>
<li>Accuracy alone becomes misleading.</li>
</ul>

<p>
For example, predicting every transaction as legitimate can achieve very high accuracy
while detecting zero fraud cases.
</p>


<hr>

<h2>3.  Objectives</h2>

<ul>

<li>
Develop a machine learning pipeline for fraud classification.
</li>

<li>
Handle severe class imbalance using advanced sampling techniques.
</li>

<li>
Compare baseline and imbalance-aware models.
</li>

<li>
Optimize fraud detection performance using appropriate evaluation metrics.
</li>

<li>
Improve detection of fraudulent transactions while controlling false alarms.
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
<td>Credit Card Transactions</td>
</tr>

<tr>
<td>Total Samples</td>
<td>284,807 transactions</td>
</tr>

<tr>
<td>Fraud Cases</td>
<td>492 transactions</td>
</tr>

<tr>
<td>Fraud Ratio</td>
<td>0.172% (Highly Imbalanced)</td>
</tr>

<tr>
<td>Target</td>
<td>Class (0: Normal, 1: Fraud)</td>
</tr>

</table>


<h3>Data Preparation</h3>

<ul>
<li>Separated features and target variable.</li>
<li>Scaled numerical features using StandardScaler.</li>
<li>Prepared training and testing datasets.</li>
<li>Applied imbalance correction only on training data.</li>
</ul>


<pre>
X = df.drop('Class', axis=1)

y = df['Class']
</pre>

<pre>

def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test, threshold=0.5):
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)

    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    # ✅ Cross-Validation ONLY on scaled training data
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = np.mean(cv_scores)

    print("Threshold:", threshold)
    print("\nLogistic Regression Classification Results:")
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


</pre>

<pre>
Baseline:
model = LogisticRegression(max_iter=1000, n_jobs=1)
model.fit(X_train_scaled, y_train)

results["Logistic"] = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)


Use: SMOTE:
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_train_scaled, y_train)

model_smote = LogisticRegression(max_iter=1000, n_jobs=1)
model_smote.fit(X_smote, y_smote)

results["SMOTE"] = evaluate_model(model_smote, X_smote, y_smote, X_test_scaled, y_test)
</pre>


<hr>

<h2>5.  Workflow</h2>


<pre>

Raw Transaction Data

        ↓

Data Cleaning & Scaling

        ↓

Baseline Logistic Regression

        ↓

Imbalance Handling

        ↓

SMOTE
Tomek Links
SMOTE-Tomek
SMOTEENN

        ↓

Model Evaluation

        ↓

Precision / Recall / F1 / PR-AUC Analysis

</pre>


<h3>Imbalance Techniques</h3>

<table border="1">

<tr>
<th>Method</th>
<th>Purpose</th>
</tr>

<tr>
<td>SMOTE</td>
<td>Synthetic minority sample generation</td>
</tr>

<tr>
<td>Tomek Links</td>
<td>Removes overlapping samples</td>
</tr>

<tr>
<td>SMOTE-Tomek</td>
<td>Combination of oversampling and cleaning</td>
</tr>

<tr>
<td>SMOTEENN</td>
<td>SMOTE with Edited Nearest Neighbor cleaning</td>
</tr>

</table>


<hr>

<h2>6. 🛠 Technologies Used</h2>

<ul>

<li><b>Programming:</b> Python</li>

<li><b>Data Processing:</b> Pandas, NumPy</li>

<li><b>Machine Learning:</b> Scikit-learn</li>

<li><b>Imbalanced Learning:</b> imbalanced-learn</li>

<li><b>Model:</b> Logistic Regression</li>

<li><b>Evaluation:</b> Confusion Matrix, Classification Report, PR-AUC</li>

<li><b>Visualization:</b> Matplotlib, Seaborn</li>

</ul>


<hr>

<h2>7.  Results</h2>


<table border="1">

<tr>
<th>Model</th>
<th>Accuracy</th>
<th>Precision</th>
<th>Recall</th>
<th>F1</th>
<th>PR-AUC</th>
</tr>


<tr>
<td>Logistic Regression</td>
<td>0.999</td>
<td>0.893</td>
<td>0.556</td>
<td>0.685</td>
<td>0.701</td>
</tr>


<tr>
<td>SMOTE</td>
<td>0.975</td>
<td>0.054</td>
<td>0.889</td>
<td>0.102</td>
<td>0.734</td>
</tr>


<tr>
<td>Tomek Links</td>
<td>0.999</td>
<td>0.897</td>
<td>0.578</td>
<td>0.703</td>
<td>0.707</td>
</tr>


<tr>
<td>SMOTEENN</td>
<td>0.974</td>
<td>0.052</td>
<td>0.889</td>
<td>0.097</td>
<td><b>0.748</b></td>
</tr>


<tr>
<td>SMOTE-Tomek</td>
<td>0.975</td>
<td>0.054</td>
<td>0.889</td>
<td>0.102</td>
<td>0.734</td>
</tr>


</table>


<img src="assets/imbalance_model_comparison.png" width="500">


<hr>

<h2>8.  Key Insights</h2>


<ul>

<li>
<b>Accuracy is misleading:</b>
Because fraud cases represent only 0.172% of transactions,
high accuracy does not guarantee fraud detection capability.
</li>


<li>
<b>Recall Optimization:</b>
SMOTE and SMOTEENN achieved the highest recall (0.889),
successfully identifying more fraudulent transactions.
</li>


<li>
<b>Precision vs Recall Trade-off:</b>
Higher recall models generated more false positives.
</li>


<li>
<b>Best Balanced Performance:</b>
Tomek Links achieved the strongest balance between Precision,
Recall, and F1-score.
</li>


<li>
<b>Best PR-AUC:</b>
SMOTEENN achieved the highest PR-AUC (0.748),
showing improved minority-class detection across thresholds.
</li>


</ul>


<h3>Final Recommendation</h3>

<p>
For financial fraud detection, the model selection depends on business priorities:
</p>

<ul>

<li>
If missing fraud cases is extremely costly:
<b>SMOTE / SMOTEENN</b> provides higher fraud detection capability.
</li>

<li>
If balanced performance is required:
<b>Tomek Links or Logistic Regression</b> provides better precision-recall balance.
</li>

</ul>


<hr>

<h2> Future Improvements</h2>

<ul>
<li>Test advanced models such as XGBoost, LightGBM, and CatBoost.</li>
<li>Apply threshold optimization instead of fixed 0.5 probability.</li>
<li>Use anomaly detection techniques.</li>
<li>Implement real-time fraud scoring API.</li>
<li>Add explainability using SHAP values.</li>
</ul>


<h2> Conclusion</h2>

<p>
This project demonstrates the importance of imbalance-aware machine learning for fraud detection.
By combining preprocessing, resampling techniques, and appropriate evaluation metrics,
the developed pipeline improves identification of fraudulent transactions while balancing
false positives and operational efficiency.
</p>