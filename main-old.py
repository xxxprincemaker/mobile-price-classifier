import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Download latest version
path = kagglehub.dataset_download("mbsoroush/mobile-price-range")

print("Path to dataset files:", path)

# Carregar o dataset
path = kagglehub.dataset_download("mbsoroush/mobile-price-range")
# Carregar o dataset
data_train = pd.read_csv(f"{path}/train.csv")
data_test = pd.read_csv(f"{path}/test.csv")

# Pré-processar os dados
X_train = data_train.drop('price_range', axis=1)
y_train = data_train['price_range']
X_test = data_test.drop('id', axis=1)

# Dividir o conjunto de treino para avaliação
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Treinar o modelo de Regressão Logística
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_split, y_train_split)
y_pred_log_reg = log_reg.predict(X_val)

# Treinar o modelo de Árvore de Decisão
tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train_split, y_train_split)
y_pred_tree = tree_clf.predict(X_val)

# Avaliar e comparar os resultados
accuracy_log_reg = accuracy_score(y_val, y_pred_log_reg)
accuracy_tree = accuracy_score(y_val, y_pred_tree)

print("Accuracy of Logistic Regression:", accuracy_log_reg)
print("Accuracy of Decision Tree:", accuracy_tree)
print("\nClassification Report for Logistic Regression:\n", classification_report(y_val, y_pred_log_reg))
print("\nClassification Report for Decision Tree:\n", classification_report(y_val, y_pred_tree))

# Prever os valores de price_range para o conjunto de teste
y_test_pred_log_reg = log_reg.predict(X_test)
y_test_pred_tree = tree_clf.predict(X_test)

print("\nPredicted price_range for test set using Logistic Regression:\n", y_test_pred_log_reg)
print("\nPredicted price_range for test set using Decision Tree:\n", y_test_pred_tree)
