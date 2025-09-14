import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 1. Load your CSV
df = pd.read_csv("test_predictions_loop_7.csv")

# 2. Create binary predictions column
df['y_pred_binary'] = (df['y_pred'] > 0.5).astype(int)

# 3. Compute metrics
y_true = df['y_true'].values
y_pred_binary = df['y_pred_binary'].values
y_pred_probs = df['y_pred'].values

accuracy = accuracy_score(y_true, y_pred_binary)
precision = precision_score(y_true, y_pred_binary)
recall = recall_score(y_true, y_pred_binary)
f1 = f1_score(y_true, y_pred_binary)
roc_auc = roc_auc_score(y_true, y_pred_probs)

# 4. Print results
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 score:  {f1:.4f}")
print(f"ROC AUC:   {roc_auc:.4f}")

# 5. (Optional) Save the updated CSV with binary column
df.to_csv("test_predictions_with_binary.csv", index=False)
