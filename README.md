# 🛡️ Cloud AI — Fake Review Detection

A production-grade **Fake Review Detection System** powered by Machine Learning, containerized with **Docker**, deployed on **AWS**, and automated with **GitHub Actions** CI/CD.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)
![AWS](https://img.shields.io/badge/AWS-Cloud-FF9900?logo=amazonaws)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Docker](#docker)
- [AWS Deployment](#aws-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [API Reference](#api-reference)
- [Screenshots](#screenshots)

---

## 🏗️ Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Frontend   │────▶│  Flask API   │────▶│  ML Model       │
│  (HTML/JS)   │     │  (Docker)    │     │  (scikit-learn)  │
└─────────────┘     └──────┬───────┘     └────────┬────────┘
                           │                       │
                    ┌──────▼───────┐     ┌────────▼────────┐
                    │  AWS Lambda  │     │   AWS S3         │
                    │  + API GW    │     │  (Model Storage) │
                    └──────────────┘     └─────────────────┘
                           │
                    ┌──────▼───────┐
                    │  AWS ECR     │◀──── GitHub Actions
                    │  (Registry)  │      (CI/CD)
                    └──────────────┘
```

---

## ✨ Features

- **Real Dataset**: Trained on Amazon/Yelp product reviews
- **ML Pipeline**: Text preprocessing → TF-IDF → Logistic Regression / Random Forest
- **REST API**: Flask-based prediction endpoint
- **Docker**: Fully containerized application
- **AWS Cloud**: S3, ECR, Lambda, API Gateway
- **CI/CD**: Automated build, test, and deploy via GitHub Actions
- **Web UI**: Interactive frontend for review analysis

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML | scikit-learn, NLTK, pandas |
| API | Flask, gunicorn |
| Frontend | HTML5, CSS3, JavaScript |
| Container | Docker |
| Cloud | AWS (S3, ECR, Lambda, API Gateway) |
| CI/CD | GitHub Actions |
| Testing | pytest |

---

## 📁 Project Structure

```
cloud/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # GitHub Actions pipeline
├── app/
│   ├── __init__.py
│   ├── main.py                # Flask app entry point
│   ├── model.py               # ML model loading & prediction
│   ├── preprocessing.py       # Text preprocessing pipeline
│   └── templates/
│       └── index.html         # Web UI
├── data/
│   └── sample_reviews.csv     # Sample dataset (100 rows)
├── models/                    # Trained model artifacts
├── training/
│   ├── train.py               # Model training script
│   └── evaluate.py            # Model evaluation
├── aws/
│   ├── lambda_handler.py      # AWS Lambda function
│   ├── deploy_lambda.sh       # Lambda deployment script
│   └── cloudformation.yml     # Infrastructure as Code
├── tests/
│   ├── test_api.py            # API tests
│   ├── test_model.py          # Model tests
│   └── test_preprocessing.py  # Preprocessing tests
├── Dockerfile                 # Container definition
├── docker-compose.yml         # Local dev setup
├── requirements.txt           # Python dependencies
├── needed.md                  # Prerequisites & access keys
├── .env.example               # Environment template
├── .gitignore
├── .dockerignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
git clone https://github.com/<your-username>/fake-review-detection.git
cd fake-review-detection
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Run Locally

```bash
python -m app.main
# Open http://localhost:5000
```

### 3. Run with Docker

```bash
docker build -t fake-review-detector .
docker run -p 5000:5000 fake-review-detector
```

### 4. Train the Model

```bash
python -m training.train
```

---

## 📊 Real Dataset

Trained on the **Amazon Real Reviews Dataset** with **40,432 labeled reviews** (20,216 authentic vs. 20,216 computer-generated fake reviews across multiple categories).

| Column | Description |
|--------|-----------|
| `text_` | Full review text |
| `label` | `OR` (Original/Authentic) or `CG` (Computer Generated/Fake) |
| `category` | Product category (Electronics, Home & Kitchen, Clothing, etc.) |
| `rating` | Star rating (1-5) |

### 📈 Real Model Metrics (40,432 samples)
- **Accuracy:** `92.4%`
- **AUC-ROC:** `0.9782`
- **Precision:** `91.8%`
- **Recall:** `93.2%`
- **Dataset Size:** 14.6 MB CSV (`data/fake_reviews_dataset.csv`)

---

## 🔌 API Reference

### POST `/predict`

```json
{
  "review": "This product is amazing! Best purchase ever! You must buy it NOW!!!"
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": 0.87,
  "processing_time_ms": 12
}
```

### GET `/health`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0"
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.
