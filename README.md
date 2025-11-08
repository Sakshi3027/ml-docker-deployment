# ML Model Deployment with Docker 

A complete machine learning model deployment project using FastAPI and Docker. This project demonstrates how to train a sentiment analysis model, create a REST API, and containerize it for easy deployment.

##  Features

- **Machine Learning Model**: Sentiment analysis using Scikit-learn
- **REST API**: FastAPI with automatic documentation
- **Containerization**: Docker for easy deployment
- **Batch Processing**: Support for single and batch predictions
- **Health Checks**: Built-in health monitoring
- **API Documentation**: Auto-generated Swagger UI

##  Project Structure

```
ml-docker-deployment/
├── app/
│   ├── __init__.py
│   └── main.py          # FastAPI application
├── models/
│   └── sentiment_model.pkl  # Trained model (generated)
├── train_model.py       # Model training script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── .dockerignore       # Docker ignore file
└── README.md          # This file
```

##  Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerization)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-docker-deployment.git
cd ml-docker-deployment
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train_model.py
```

This will create a `sentiment_model.pkl` file in the `models/` directory.

### 4. Run the API Locally

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t ml-sentiment-api .
```

### Run Docker Container

```bash
docker run -d -p 8000:8000 --name ml-api ml-sentiment-api
```

### Check Container Status

```bash
docker ps
docker logs ml-api
```

### Stop Container

```bash
docker stop ml-api
docker rm ml-api
```

##  API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This product is amazing!"}'
```

**Response:**
```json
{
  "text": "This product is amazing!",
  "sentiment": "Positive",
  "confidence": 0.8542,
  "timestamp": "2025-11-08T10:30:00"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{
       "texts": [
         "I love this!",
         "This is terrible",
         "Pretty good overall"
       ]
     }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

##  Testing

### Using Python

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "This is fantastic!"}
)
print(response.json())

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Great!", "Terrible!", "Not bad"]}
)
print(response.json())
```

### Using JavaScript

```javascript
// Single prediction
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({text: 'This is amazing!'})
})
.then(res => res.json())
.then(data => console.log(data));
```

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI documentation |
| POST | `/predict` | Single text prediction |
| POST | `/predict/batch` | Batch text prediction |

##  Technology Stack

- **Python 3.11**: Programming language
- **Scikit-learn**: Machine learning library
- **FastAPI**: Modern web framework
- **Uvicorn**: ASGI server
- **Docker**: Containerization platform
- **Pydantic**: Data validation

##  Model Information

- **Type**: Sentiment Analysis (Binary Classification)
- **Algorithm**: Naive Bayes with TF-IDF vectorization
- **Classes**: Positive, Negative
- **Features**: TF-IDF with unigrams and bigrams

##  Configuration

### Environment Variables

You can configure the application using environment variables:

```bash
export MODEL_PATH=models/sentiment_model.pkl
export PORT=8000
```

### Docker Compose (Optional)

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  ml-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH=models/sentiment_model.pkl
```

Run with: `docker-compose up -d`

##  Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

##  License

This project is licensed under the MIT License.

## 👤 Author

**Sakshi**
- GitHub: [Sakshi3027](https://github.com/Sakshi3027)

## Acknowledgments

- FastAPI documentation
- Scikit-learn community
- Docker documentation

