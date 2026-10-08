#!/bin/bash
# Deploy Lambda function and API Gateway
# Usage: ./aws/deploy_lambda.sh

set -e

# Configuration
FUNCTION_NAME="fake-review-detector"
REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET="${S3_BUCKET_NAME}"
RUNTIME="python3.10"
HANDLER="aws.lambda_handler.lambda_handler"
TIMEOUT=30
MEMORY=512

echo "=========================================="
echo "  Deploying Fake Review Detector Lambda"
echo "=========================================="

# Step 1: Package the Lambda function
echo "[1/5] Packaging Lambda function..."
mkdir -p /tmp/lambda_package
pip install -r requirements.txt -t /tmp/lambda_package --quiet

# Copy application code
cp -r app/ /tmp/lambda_package/
cp -r aws/ /tmp/lambda_package/
cp -r models/ /tmp/lambda_package/ 2>/dev/null || echo "[WARN] No models directory found"

# Create deployment package
cd /tmp/lambda_package
zip -r /tmp/lambda_deployment.zip . -q
cd -
echo "[INFO] Package size: $(du -h /tmp/lambda_deployment.zip | cut -f1)"

# Step 2: Upload model to S3
echo "[2/5] Uploading model to S3..."
if [ -f "models/fake_review_model.pkl" ]; then
    aws s3 cp models/fake_review_model.pkl s3://${S3_BUCKET}/models/fake_review_model.pkl
    echo "[INFO] Model uploaded to S3"
else
    echo "[WARN] No model file found. Upload manually after training."
fi

# Step 3: Create or update Lambda function
echo "[3/5] Deploying Lambda function..."
if aws lambda get-function --function-name $FUNCTION_NAME --region $REGION 2>/dev/null; then
    echo "[INFO] Updating existing function..."
    aws lambda update-function-code \
        --function-name $FUNCTION_NAME \
        --zip-file fileb:///tmp/lambda_deployment.zip \
        --region $REGION
else
    echo "[INFO] Creating new function..."
    # Create execution role if it doesn't exist
    ROLE_ARN=$(aws iam get-role --role-name lambda-fake-review-role --query 'Role.Arn' --output text 2>/dev/null || true)

    if [ -z "$ROLE_ARN" ]; then
        echo "[INFO] Creating IAM role..."
        ROLE_ARN=$(aws iam create-role \
            --role-name lambda-fake-review-role \
            --assume-role-policy-document '{
                "Version": "2012-10-17",
                "Statement": [{
                    "Effect": "Allow",
                    "Principal": {"Service": "lambda.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }]
            }' \
            --query 'Role.Arn' --output text)

        # Attach policies
        aws iam attach-role-policy --role-name lambda-fake-review-role \
            --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
        aws iam attach-role-policy --role-name lambda-fake-review-role \
            --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

        echo "[INFO] Waiting for role propagation..."
        sleep 10
    fi

    aws lambda create-function \
        --function-name $FUNCTION_NAME \
        --runtime $RUNTIME \
        --handler $HANDLER \
        --role $ROLE_ARN \
        --zip-file fileb:///tmp/lambda_deployment.zip \
        --timeout $TIMEOUT \
        --memory-size $MEMORY \
        --environment "Variables={S3_BUCKET_NAME=$S3_BUCKET}" \
        --region $REGION
fi

# Step 4: Update environment variables
echo "[4/5] Updating environment variables..."
aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment "Variables={S3_BUCKET_NAME=$S3_BUCKET}" \
    --region $REGION \
    --timeout $TIMEOUT \
    --memory-size $MEMORY 2>/dev/null || true

# Step 5: Create API Gateway (if not exists)
echo "[5/5] Setting up API Gateway..."
API_ID=$(aws apigateway get-rest-apis --region $REGION \
    --query "items[?name=='FakeReviewAPI'].id" --output text)

if [ -z "$API_ID" ] || [ "$API_ID" = "None" ]; then
    echo "[INFO] Creating API Gateway..."
    API_ID=$(aws apigateway create-rest-api \
        --name "FakeReviewAPI" \
        --description "Fake Review Detection API" \
        --region $REGION \
        --query 'id' --output text)
    echo "[INFO] API Gateway created: $API_ID"
else
    echo "[INFO] API Gateway already exists: $API_ID"
fi

echo ""
echo "=========================================="
echo "  ✅ Deployment Complete!"
echo "=========================================="
echo "  Function: $FUNCTION_NAME"
echo "  Region:   $REGION"
echo "  API ID:   $API_ID"
echo "=========================================="

# Cleanup
rm -rf /tmp/lambda_package /tmp/lambda_deployment.zip
