import requests
import os

# Test case 1: Test the /checker endpoint with existing model and vectorizer
def test_predict_existing_model():
    data = {
        "password": ["password123", "123456", "qwerty", "Qb86c42!as38v97b.", "P@ssw0rd!"]
    }
    response = requests.post("http://localhost:5000/checker", json=data)
    predictions = response.json()
    assert predictions == [0, 0, 0, 2, 1]

# Test case 2: Test the /checker endpoint without existing model and vectorizer
def test_predict_new_model():
    # Remove existing model and vectorizer files
    if os.path.exists('model.joblib'):
        os.remove('model.joblib')
    if os.path.exists('vectorizer.joblib'):
        os.remove('vectorizer.joblib')

    data = {
        "password": ["password123", "123456", "qwerty"]
    }
    response = requests.post("http://localhost:5000/checker", json=data)
    predictions = response.json()
    assert predictions == [0, 0, 0]

# Test case 3: Test the /checker endpoint with invalid data
def test_predict_invalid_data():
    data = {
        "passwords": ["password123", "123456", "qwerty"]  # Invalid key name
    }
    response = requests.post("http://localhost:5000/checker", json=data)
    predictions = response.json()
    assert predictions == {'error': "'password'"}

# Test case 4: Test the /generate endpoint with valid parameters
def test_generate_password():
    data = {
        "acrostic": "hello",
        "delimiter": "-",
        "min_length": 8,
        "max_length": 12
    }
    response = requests.post("http://localhost:5000/generate", json=data)
    generated_password = response.json()
    assert 'password' in generated_password

# Test case 2: Test the /generate endpoint with missing parameters
def test_generate_password_missing_parameters():
    data = {
        "acrostic": "hello",
        "min_length": 8,
        "max_length": 12
    }
    response = requests.post("http://localhost:5000/generate", json=data)
    generated_password = response.json()
    assert generated_password == {'error': 'Missing parameters in request'}

# Test case 3: Test the /generate endpoint with invalid parameters
def test_generate_password_invalid_parameters():
    data = {
        "acrostic": "hello",
        "delimiter": "-",
        "min_length": "8",
        "max_length": "12"
    }
    response = requests.post("http://localhost:5000/generate", json=data)
    generated_password = response.json()
    assert generated_password == {'error': 'Invalid parameter types'}

# Run the tests
test_predict_existing_model()
test_predict_new_model()
test_predict_invalid_data()
test_generate_password()
test_generate_password_missing_parameters()
test_generate_password_invalid_parameters()