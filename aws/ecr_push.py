"""
Helper script to authenticate Docker with AWS ECR and push the image.
"""

import os
import base64
import subprocess
import boto3
from dotenv import load_dotenv

load_dotenv()


def push_to_ecr():
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    region = os.getenv('AWS_DEFAULT_REGION', 'ap-southeast-2')
    account_id = os.getenv('AWS_ACCOUNT_ID', '948897434480')
    repo_name = 'fake-review-detector'

    ecr = boto3.client('ecr', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret, region_name=region)

    print(f"[*] Getting ECR authorization token for region {region}...")
    auth_data = ecr.get_authorization_token()['authorizationData'][0]
    token = base64.b64decode(auth_data['authorizationToken']).decode('utf-8')
    username, password = token.split(':')
    endpoint = auth_data['proxyEndpoint']

    print(f"[*] Logging in Docker to ECR: {endpoint}...")
    res = subprocess.run(
        ['docker', 'login', '-u', username, '--password-stdin', endpoint],
        input=password,
        text=True,
        capture_output=True
    )
    print(res.stdout)
    if res.returncode != 0:
        print("[!] Docker login error:", res.stderr)
        return False

    image_tag = f"{account_id}.dkr.ecr.{region}.amazonaws.com/{repo_name}:latest"
    print(f"[*] Pushing image {image_tag} to AWS ECR...")
    push_res = subprocess.run(['docker', 'push', image_tag], text=True)
    if push_res.returncode == 0:
        print(f"[+] Successfully pushed to AWS ECR: {image_tag}")
        return True
    else:
        print("[!] Push to AWS ECR failed.")
        return False


if __name__ == '__main__':
    push_to_ecr()
