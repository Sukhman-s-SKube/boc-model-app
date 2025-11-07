from os import getenv
import boto3

AWS_ENDPOINT_URL = getenv("AWS_ENDPOINT_URL")
AWS_REGION = getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_DEFAULT = getenv("S3_BUCKET", "dagster")
MODELS_BUCKET_DEFAULT = getenv("MODELS_BUCKET", "models")

def get_s3():
    S3 = boto3.client(
        "s3",
        endpoint_url=AWS_ENDPOINT_URL,
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    return S3




