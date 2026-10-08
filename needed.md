# 🔐 NEEDED — Prerequisites & Access Keys

> **Project:** Cloud AI for Fake Review Detection  
> **Stack:** Python · Docker · AWS · GitHub Actions  
> **Status:** ⬜ = Not Done · ✅ = Done

---

## 1. Accounts Required

| # | Account | Sign-Up Link | Status |
|---|---------|-------------|--------|
| 1 | **GitHub** (free) | https://github.com/signup | ⬜ |
| 2 | **AWS** (free tier eligible) | https://aws.amazon.com/free | ⬜ |
| 3 | **Docker Hub** (free) | https://hub.docker.com/signup | ⬜ |

---

## 2. AWS Credentials & Services

### 2.1 IAM User (Programmatic Access)

> **IMPORTANT:** Create a dedicated IAM user — never use your root account keys.

1. Go to **AWS Console → IAM → Users → Add User**
2. User name: `fake-review-detector`
3. Access type: Programmatic access
4. Attach policies:
   - `AmazonS3FullAccess`
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonSageMakerFullAccess`
   - `AWSLambda_FullAccess`
   - `CloudWatchFullAccess`
5. Download the credentials CSV

| Key | Value | Status |
|-----|-------|--------|
| `AWS_ACCESS_KEY_ID` | `AKIA5Z3VRZNYFMTEXA62` | ✅ Configured |
| `AWS_SECRET_ACCESS_KEY` | `UJmZ...` (in .env) | ✅ Configured |
| `AWS_DEFAULT_REGION` | `ap-southeast-2` | ✅ Verified |
| `AWS_ACCOUNT_ID` | `948897434480` | ✅ Verified |
| `S3_BUCKET_NAME` | `fake-review-detector` | ✅ Created & Active |
| `ECR_REPOSITORY_URI` | `948897434480.dkr.ecr.ap-southeast-2.amazonaws.com/fake-review-detector` | ✅ Created & Active |
| `DOCKER_USERNAME` | `parthb666` | ✅ Ready |
| `GITHUB_REPO` | `https://github.com/Parthx-06/fake_reviews.git` | ✅ Linked |

### 2.2 AWS Services Used

| Service | Purpose | Free Tier? |
|---------|---------|-----------|
| **S3** | Store dataset & model artifacts | 5 GB |
| **ECR** | Docker container registry | 500 MB |
| **SageMaker** | Model training & endpoint | 250 hrs |
| **Lambda** | Serverless inference API | 1M requests/month |
| **API Gateway** | REST API for the Lambda | 1M calls/month |
| **CloudWatch** | Logging & monitoring | Basic |

### 2.3 S3 Bucket

```bash
aws s3 mb s3://fake-review-detection-<your-unique-id> --region us-east-1
```

| Key | Value | Status |
|-----|-------|--------|
| `S3_BUCKET_NAME` | `fake-review-detection-<your-unique-id>` | ⬜ |

### 2.4 ECR Repository

```bash
aws ecr create-repository --repository-name fake-review-detector --region us-east-1
```

| Key | Value | Status |
|-----|-------|--------|
| `ECR_REPOSITORY_URI` | `<account-id>.dkr.ecr.us-east-1.amazonaws.com/fake-review-detector` | ⬜ |
| `AWS_ACCOUNT_ID` | `123456789012` | ⬜ |

---

## 3. Docker Hub Credentials

| Key | Value | Status |
|-----|-------|--------|
| `DOCKER_USERNAME` | your Docker Hub username | ⬜ |
| `DOCKER_PASSWORD` | your Docker Hub password or access token | ⬜ |

> Use a **Docker Hub Access Token** instead of your password.
> Go to Docker Hub → Account Settings → Security → New Access Token.

---

## 4. GitHub Secrets (for CI/CD)

Go to **GitHub Repo → Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Value | Status |
|-------------|-------|--------|
| `AWS_ACCESS_KEY_ID` | From IAM user | ⬜ |
| `AWS_SECRET_ACCESS_KEY` | From IAM user | ⬜ |
| `AWS_REGION` | `us-east-1` | ⬜ |
| `AWS_ACCOUNT_ID` | Your 12-digit AWS account ID | ⬜ |
| `S3_BUCKET_NAME` | Your S3 bucket name | ⬜ |
| `DOCKER_USERNAME` | Docker Hub username | ⬜ |
| `DOCKER_PASSWORD` | Docker Hub access token | ⬜ |

---

## 5. Dataset (Real Data)

This project uses the **Amazon Product Reviews** dataset.

| Item | Details |
|------|---------|
| **Dataset** | Amazon Product Reviews (Labeled Fake/Real) |
| **Source** | https://www.kaggle.com/datasets/lievgarcia/amazon-reviews |
| **Alternative** | https://www.yelp.com/dataset |
| **Format** | CSV |

### Kaggle API (Optional)

| Key | Value | Status |
|-----|-------|--------|
| `KAGGLE_USERNAME` | your Kaggle username | ⬜ |
| `KAGGLE_KEY` | your Kaggle API key | ⬜ |

> Get your API key from: https://www.kaggle.com/settings → API → Create New Token

---

## 6. Local Development Tools

| Tool | Version | Install Link | Status |
|------|---------|-------------|--------|
| **Python** | 3.10+ | https://python.org/downloads | ⬜ |
| **Docker Desktop** | Latest | https://docker.com/products/docker-desktop | ⬜ |
| **AWS CLI v2** | Latest | https://aws.amazon.com/cli/ | ⬜ |
| **Git** | Latest | https://git-scm.com/downloads | ⬜ |

### Configure AWS CLI

```bash
aws configure
# Enter: AWS_ACCESS_KEY_ID
# Enter: AWS_SECRET_ACCESS_KEY
# Enter: us-east-1
# Enter: json
```

---

## 7. Environment Variables (.env file)

Create a `.env` file in the project root:

```env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
S3_BUCKET_NAME=fake-review-detection-yourname
DOCKER_USERNAME=your_docker_username
FLASK_ENV=development
MODEL_PATH=models/fake_review_model.pkl
```

> **CAUTION:** NEVER commit your `.env` file to Git.

---

## 8. Estimated AWS Costs (Free Tier)

| Service | Free Tier Limit | Cost Beyond Free Tier |
|---------|----------------|----------------------|
| S3 | 5 GB storage | ~$0.023/GB/month |
| ECR | 500 MB storage | ~$0.10/GB/month |
| SageMaker | 250 hrs t2.medium | ~$0.05/hr |
| Lambda | 1M requests | ~$0.20 per 1M |
| API Gateway | 1M calls | ~$3.50 per 1M |

---

## Quick Verification Checklist

- [ ] GitHub account created & repo initialized
- [ ] AWS account created & IAM user configured
- [ ] AWS CLI installed & configured locally
- [ ] Docker Desktop installed & running
- [ ] Docker Hub account created
- [ ] S3 bucket created
- [ ] ECR repository created
- [ ] .env file created with all values
- [ ] GitHub Secrets configured
- [ ] Dataset downloaded
- [ ] Python 3.10+ installed
- [ ] All pip dependencies installed
