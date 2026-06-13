from fastapi.testclient import TestClient
from API.main import app  # karena folder API huruf besar

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Selamat datang" in response.json()["message"]

def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "aplikasi ini sangat membantu"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positif", "negatif"]
    assert 0 <= data["confidence"] <= 1

def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "aplikasi jelek dan sering error"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] in ["positif", "negatif"]
    assert 0 <= data["confidence"] <= 1