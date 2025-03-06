import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Assuming you have already loaded your model, X_test_images, X_test_tabular, and y_test

#Evalution of model
def evaluate(model, X_test_images, X_test_tabular, y_test):
    # Evaluate the model
    loss, accuracy = model.evaluate([X_test_images, X_test_tabular], y_test, verbose=0)
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    # Get the model's predictions
    y_pred_probs = model.predict([X_test_images, X_test_tabular])

    # Convert probabilities to predicted labels (binary classification)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Handle the case where y_test might be one-hot encoded (unlikely in binary, but checking)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
    else:
        y_test_labels = y_test

    # Calculate precision, recall, and F1 score (binary classification)
    precision = precision_score(y_test_labels, y_pred)
    recall = recall_score(y_test_labels, y_pred)
    f1score = f1_score(y_test_labels, y_pred)

    # Print the results
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1score:.4f}')
