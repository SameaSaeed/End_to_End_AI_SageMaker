import sys
import sagemaker
from sagemaker import get_execution_role
from sagemaker.model import Model
from sagemaker import image_uris

# Get command-line arguments
if len(sys.argv) != 3:
    print("Usage: python deploy_onnx.py <s3_onnx_model_uri> <model_name>")
    sys.exit(1)

onnx_model_s3_uri = sys.argv[1]  # e.g., s3://bucket/folder/evaluation/model.onnx
model_name = sys.argv[2]         # SageMaker model name

# Set up environment
sagemaker_session = sagemaker.Session()
role = get_execution_role()

def deploy_onnx_model(onnx_model_s3_uri, model_name):
    """Deploy an ONNX model from S3 to a SageMaker endpoint"""
    model = Model(
        image_uri=image_uris.retrieve(
            framework="onnx",
            region=sagemaker_session.boto_region_name,
            version="1.12.0",  # ONNX Inference image version
            py_version="py38",
        ),
        model_data=onnx_model_s3_uri,
        role=role,
        sagemaker_session=sagemaker_session,
        name=model_name
    )

    endpoint = model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large"
    )
    return endpoint

# Deploy the model
endpoint = deploy_onnx_model(onnx_model_s3_uri, model_name)
print(f"Model deployed at endpoint: {endpoint.endpoint_name}")
