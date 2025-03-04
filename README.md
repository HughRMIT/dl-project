# dl-project
## Deep Learning Project - A complilation of my deep learning python scripts.
This code defines a deep learning model for text classification using pre-trained embeddings and bidirectional LSTM layers. I've integrated Keras Tuner to search for optimal hyperparameters.

Pre-requisites:
Python 3

Ensure you have the following Python packages installed:
tensorflow
keras_tuner
sklearn

You can install them using pip:
pip install tensorflow keras-tuner scikit-learn

Running the Code:
Before using the model, ensure that you have:
A vocabulary size (vocab_size) defined
The embedding dimensions (embedding_dim) set
A pre-trained embedding_matrix

Hyperparameter Tuning:
Start by initialising the Keras Tuner's RandomSearch or any other tuner and then search for the best hyperparameters. Make sure to define the search space (e.g., number of trials, epochs).

Compute Class Weights:
To balance the training for imbalanced classes, compute class weights based on your training labels:
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train_labels), y_train_labels)
class_weights = dict(enumerate(class_weights))

Remember to convert one-hot encoded labels back to label encoding if necessary.

Training the Model:
Once you have the best hyperparameters, build the model using build_model() and train using the model.fit() method, specifying the class_weight argument.

Evaluation:
Use the model.evaluate() function on your test dataset to see how well the model performs.

Notes:
Monitor the model's performance, not just based on accuracy but other metrics such as precision, recall, and F1-score.
Adjust the model's architecture or hyperparameters based on the results.
Ensure that you have sufficient computational resources, as deep learning models, especially with large vocabularies, can be resource-intensive.
Alternatively, the model can be trained on Google Colab or Kaggle, which provides free GPU resources and can be accessed through a web browser.
Simply, import the Python notebook, ipynb file, and run it.
