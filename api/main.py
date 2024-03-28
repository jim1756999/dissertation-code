import string
import numpy as np
import pandas as pd
import os
import random
import seaborn as sns
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify
from flask_cors import CORS
from xkcdpass import xkcd_password as xkcd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


app = Flask(__name__)
cors = CORS(app)

def test_model(model, X_test, y_test):
    """
    Test the model and print the accuracy score.

    Args:
        model: The trained model.
        X_test: The test data.
        y_test: The test labels.
    """
    # Predict the test data
    y_pred = model.predict(X_test)

    # Calculate the accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Plot the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    xgb.plot_tree(model, num_trees=0)

    residuals = [true - pred for true, pred in zip(y_test, y_pred)]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, color='blue', edgecolor='k')
    plt.axhline(y=0, color='red', linestyle='--')  
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.residplot(x=y_pred, y=y_test, lowess=True, color="g", line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.show()


def plot_learning_curve(estimator, X_train, y_train, X, y, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Plot the learning curve.

    Args:
    - estimator: The model to be trained.
    - X_train: The features of the training dataset.
    - y_train: The labels of the training dataset.
    - X: The features of the entire dataset, used for cross-validation.
    - y: The labels of the entire dataset, used for cross-validation.
    - train_sizes: An array of training set sizes used to generate the learning curve.
    """
    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=10, n_jobs=-1, train_sizes=train_sizes, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def load_and_preprocess_data(filepath):
    """
    Load and preprocess data from a CSV file.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The preprocessed data.
    """
    data = pd.read_csv(filepath, encoding="ISO-8859-1", on_bad_lines='skip')
    data.dropna(inplace=True)
    return data


def shuffle_data(data):
    """
    Shuffle the rows of a numpy array.

    Args:
        data (numpy.ndarray): The input data.

    Returns:
        numpy.ndarray: The shuffled data.
    """
    data_array = np.array(data)
    random.shuffle(data_array)
    return data_array


def analyse_dataset(data):
    """
    Plot the distribution of data.

    Args:
        data (pandas.DataFrame): The input data.
    """
    print(data.head())
    print(data.isnull().sum())
    sns.set_style('whitegrid')
    sns.countplot(x='strength', data=data)


def word_char(inputs):
    """
    Tokenizer function for vectorization.

    Args:
        inputs (str): The input string.

    Returns:
        list: The list of characters in the input string.
    """
    return [char for char in inputs]


def create_model(analysis=False):
    """
    Create and train a machine learning model.

    Returns:
        tuple: A tuple containing the trained model and the vectorizer.
    """
    # Load data
    data = load_and_preprocess_data("./data.csv")

    # Data analysis

    # Prepare data for modeling
    shuffled_data = shuffle_data(data)
    y = [labels[1] for labels in shuffled_data]
    x = [labels[0] for labels in shuffled_data]

    # Vectorize x
    vect = TfidfVectorizer(tokenizer=word_char)
    x = vect.fit_transform(x)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=18)

    # Train model
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)

    # Evaluate model
    score = model.score(X_test, y_test)
    print(f"Model Score: {score}")

    if analysis:
        analyse_dataset(data)
        test_model(model, X_test, y_test)
        # plot_learning_curve(model, X_train, y_train, x, y).show()

    # Save model
    dump(model, 'model.joblib')
    dump(vect, 'vectorizer.joblib')

    return model, vect


def read_model():
    """
    Load a pre-trained model and vectorizer.

    Returns:
        tuple: A tuple containing the loaded model and vectorizer.
    """
    # Load model
    model = load('model.joblib')
    vect = load('vectorizer.joblib')

    return model, vect

def predict_password_strength(model, vect, test_texts):
    """
    Predict the strength of a password.

    Args:
        model: The trained model.
        vect: The vectorizer.
        test_texts: The passwords to predict.

    Returns:
        list: The predicted strengths.
    """
    try:
        # Transform texts with the loaded vectorizer
        transformed_texts = vect.transform(test_texts)

        # Predict with the loaded model
        predictions = model.predict(transformed_texts)

        # Return predictions
        return predictions.tolist()

    except Exception as e:
        return {'error': str(e)}
    
def basic_checks(password):
    """Perform basic password strength validations."""
    if len(password) < 8:
        return False  # Too short
    if not any(char.islower() for char in password):
        return False  # No lowercase letter
    if not any(char.isupper() for char in password):
        return False  # No uppercase letter
    if not any(char.isdigit() for char in password):
        return False  # No digit
    return True


def randomize_case(word):
    # Randomly change the case of each letter in the word
    return ''.join(random.choice([letter.lower(), letter.upper()]) for letter in word)

def generate(acrostic, delimiter, min_length, max_length):
    """
    Generate a password using XKCD wordlist.

    Args:
        acrostic (str): The acrostic pattern for the password.
        delimiter (str): The delimiter to separate words in the password.
        min_length (int): The minimum length of each word in the password.
        max_length (int): The maximum length of each word in the password.

    Returns:
        str: The generated password.

    """ 
    wordlist = xkcd.generate_wordlist(wordfile=None, min_length=min_length, max_length=max_length)

    generated_password_words_list = []
    for acrostic_in_letter in acrostic:
        if acrostic_in_letter not in string.ascii_letters:
            return jsonify({'error': 'Invalid acrostic pattern'}), 400
        print("LETTER:" + acrostic_in_letter)
        generated_password_words = xkcd.generate_xkcdpassword(acrostic=acrostic_in_letter, wordlist=wordlist, delimiter=delimiter)
        generated_password_words_list.append(generated_password_words + "-")

    words = generated_password_words_list

        
    # words = []
    # for i in range(5):
    #     gnt = xkcd.generate_xkcdpassword(wordlist, delimiter=delimiter, numwords=1)
    #     print(gnt)
    #     words.append(gnt)
    #     print(words)
    #     i += 1
    # selected_words = [randomize_case(words) for words in random.sample(words, len(words))]
    selected_words = [randomize_case(words) for words in words]
    digits = ''.join(random.choice(string.digits) for i in range(3))
    symbols = ''.join(random.choice(string.punctuation) for i in range(3))
    password = ''.join(selected_words) + digits + symbols

 
    return password



@app.route('/generate', methods=['POST'])
def generate_password():
    """
    Generate a password based on the given parameters.

    Parameters:
    - acrostic (str): The acrostic string to generate the password from.
    - delimiter (str): The delimiter to separate the characters in the acrostic string.
    - min_length (int): The minimum length of the generated password.
    - max_length (int): The maximum length of the generated password.

    Returns:
    - response (dict): A dictionary containing the generated password.

    Example Usage:
    ```
    {
        "acrostic": "hello",
        "delimiter": "-",
        "min_length": 8,
        "max_length": 12
    }
    ```
    """
    data = request.get_json()

    if not all(key in data for key in ['acrostic', 'delimiter', 'min_length', 'max_length']):
        return jsonify({'error': 'Missing parameters in request'}), 400

    acrostic = data['acrostic']
    delimiter = data['delimiter']
    min_length = data['min_length']
    max_length = data['max_length']

    if not isinstance(acrostic, str) or not isinstance(delimiter, str) or not isinstance(min_length, int) or not isinstance(max_length, int):
        return jsonify({'error': 'Invalid parameter types'}), 400

    response = {
        'password': generate(acrostic, delimiter, min_length, max_length)
    }

    return jsonify(response)


@app.route('/checker', methods=['POST'])
def checker():
    """
    Endpoint for checking the strength of passwords.

    This function receives a POST request with a JSON payload containing a list of passwords.
    It performs basic checks on each password and predicts their strength using a pre-trained model.
    The predicted strength scores are returned as a JSON response.

    Returns:
        A JSON response containing the predicted strength scores for the passwords.

    Raises:
        Exception: If an error occurs during the processing of the request.
    """
    try:
        if os.path.exists('model.joblib') and os.path.exists('vectorizer.joblib'):
            model, vect = read_model()
        else:
            model, vect = create_model()

        # Get data from request
        data = request.json
        passwords = data['password']

        # Store results
        results = []

        for password in passwords:
            if basic_checks(password):
                # If password passes basic checks, use the model to predict strength
                prediction = predict_password_strength(model, vect, [password])
            else:
                # If password fails basic checks, assign a low strength score
                prediction = [0] # Low strength

            results.append(prediction[0])

        # Return predictions as a JSON response
        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)})


# Main execution
if __name__ == "__main__":
    app.run(debug=True, port=5000)