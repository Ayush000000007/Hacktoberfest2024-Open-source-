import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have a function to load your data and model
# y_true should be your actual labels, and y_probas should be your model's predictions
y_true = []  # Replace with actual loading code
y_probas = []  # Replace with actual model predictions code

# Debugging: Print y_true and y_probas directly after loading
print("Loaded y_true:", y_true)
print("Generated y_probas:", y_probas)

# Convert to numpy arrays if not already
y_true = np.array(y_true)
y_probas = np.array(y_probas)

# Ensure they are not empty
if y_true.size == 0 or y_probas.size == 0:
    print("Error: One of the arrays is empty. Check your data loading and model prediction steps.")
else:
    # Calculate Precision-Recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_probas, pos_label=1)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)

    print(f"Optimal Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores[:-1], label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Threshold")
    plt.legend()
    plt.grid()
    plt.show()
