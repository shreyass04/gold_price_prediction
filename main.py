import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# 1. Load dataset
gold_data = pd.read_csv("gld_price_data.csv")

# 2. Show first 5 rows
print("First 5 rows:\n", gold_data.head())

# 3. Check missing values
print("\nMissing values:\n", gold_data.isnull().sum())

# 4. Correlation (remove Date)
correlation = gold_data.drop(['Date'], axis=1).corr()

# 5. Plot heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8})
plt.title("Correlation Heatmap")
plt.show()

# 6. Input (X) and Output (Y)
X = gold_data.drop(['Date', 'GLD'], axis=1)
Y = gold_data['GLD']

# 7. Train-Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# 8. Model
model = RandomForestRegressor(n_estimators=100)

# 9. Train model
model.fit(X_train, Y_train)

# 10. Prediction
test_data_prediction = model.predict(X_test)

# 11. Accuracy
error_score = metrics.r2_score(Y_test, test_data_prediction)
print("\nR2 Score:", error_score)

# 12. Compare actual vs predicted
plt.figure(figsize=(10,5))
plt.plot(Y_test.values[:100], label='Actual')
plt.plot(test_data_prediction[:100], label='Predicted')
plt.title("Actual vs Predicted Gold Price")
plt.legend()
plt.show()