
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, classification_report, \
    RocCurveDisplay
from sklearn.metrics import confusion_matrix
from utils import load_data
import matplotlib.pyplot as plt
import os
import pickle

# Load dataset and labels
X_train, X_test, y_train, y_test = load_data(test_size=0.25)

print("[+] Number of training samples:", X_train.shape[0])
print("[+] Number of testing samples:", X_test.shape[0])
print("[+] Number of features:", X_train.shape[1])

# Configure parameter space to search
mlp_gs = MLPClassifier(max_iter=50000)
parameter_space = {
    'hidden_layer_sizes': [(35,), (50,), (20,)],
    'activation': ['tanh', 'relu'],
    'solver': ['adam'],
    'epsilon': [1e-08, 1e-04, 1],
    'alpha': [0.0001, 0.05, 0.1],
    'learning_rate': ['constant', 'adaptive'],
}

print("[*] Training the model...")

# Perform grid search
model = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
model.fit(X_train, y_train)  # X is train samples and y is the corresponding labels

# Output grid search results
print('Best parameters found: ', model.best_params_)

# Test over testing data and output accuracy
y_true, y_pred = y_test, model.predict(X_test)
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Make result directory if it does not exist
if not os.path.exists("external_programs/result"):
    os.makedirs("external_programs/result")

# Dump model to file for future use
pickle.dump(model, open("external_programs/result/mlp_classifier.model", "wb"))

# Output analytics
confusion_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=model.classes_)
disp.plot()
plt.savefig("external_programs/result/confusion_matrix.jpg")

# Save classification report to a file
with open("external_programs/result/classification_report.txt", "w") as f:
    f.write(classification_report(y_true=y_test, y_pred=y_pred))

# Save best parameters to a file
with open("external_programs/result/best_parameters.txt", "w") as f:
    f.write(str(model.best_params_))

# Save accuracy to a file
with open("external_programs/result/accuracy.txt", "w") as f:
    f.write(str(accuracy))
