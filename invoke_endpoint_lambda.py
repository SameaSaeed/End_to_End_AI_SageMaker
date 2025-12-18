import json
import boto3
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from urllib.parse import urlparse

# Initialize clients
s3 = boto3.client('s3')
sagemaker_runtime = boto3.client('sagemaker-runtime')

def preprocess_image(base64_image, target_size=(640, 640)):
    """Decode base64 image and convert to normalized tensor for ONNX"""
    image_bytes = base64.b64decode(base64_image)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize(target_size)
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32) / 255.0
    # HWC to CHW
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.tolist()  # Convert to list for JSON serialization

def lambda_handler(event, context):
    try:
        body = json.loads(event['body'])
        s3_uri = body['s3_uri']
        endpoint_name = body['endpoint_name']
        input_data = body['input_data']
        thresholds = body.get('thresholds', {})

        # Parse S3 URI
        parsed_uri = urlparse(s3_uri)
        bucket = parsed_uri.netloc
        key_prefix = parsed_uri.path.lstrip('/')

        # Download metrics
        metrics_key = f"{key_prefix}/metrics.json"
        local_metrics_path = '/tmp/metrics.json'
        s3.download_file(bucket, metrics_key, local_metrics_path)
        with open(local_metrics_path, 'r') as f:
            metrics = json.load(f)

        # Check required metrics
        required_keys = ['mAP', 'mAP50', 'mAP75', 'precision', 'recall']
        if not all(k in metrics for k in required_keys):
            return {'statusCode': 400, 'body': json.dumps({'result': False, 'error': 'Missing metrics'})}

        # Preprocess input image
        img_tensor = preprocess_image(input_data['image'])

        # Invoke SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=json.dumps({"input": img_tensor})
        )
        result = json.loads(response['Body'].read().decode())

        # Compare thresholds
        meets_criteria = True
        for key, threshold in thresholds.items():
            if result.get(key, float('-inf')) < threshold:
                meets_criteria = False
                break

        return {
            'statusCode': 200,
            'body': json.dumps({
                'result': meets_criteria,
                'metrics': result
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'result': False, 'error': str(e)})
        }
