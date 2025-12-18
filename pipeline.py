import sagemaker
from sagemaker import get_execution_role
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString, ParameterInteger, ParameterFloat
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SageMaker environment
SAGEMAKER_SESSION = sagemaker.Session()
ROLE = get_execution_role()
PIPELINE_SESSION = sagemaker.workflow.PipelineSession()

# Parameters
s3_bucket_param = ParameterString("S3Bucket", default_value="yolo-wildfire-smoke-detection")
s3_folder_param = ParameterString("S3Folder", default_value="wildfire_smoke")
model_param = ParameterString("Model", default_value="yolov10n.pt")
epochs_param = ParameterInteger("Epochs", default_value=10)
batch_size_param = ParameterInteger("BatchSize", default_value=16)

# Images
PROCESSING_IMAGE = retrieve("pytorch", SAGEMAKER_SESSION.boto_region_name, version="2.2.0", py_version="py310", image_scope="training")
TRAINING_IMAGE = PROCESSING_IMAGE

# Preprocessing step
preprocess_step = ProcessingStep(
    name="YOLO-Preprocessing",
    processor=ScriptProcessor(
        image_uri=PROCESSING_IMAGE,
        command=["python3"],
        role=ROLE,
        instance_count=1,
        instance_type="ml.t3.xlarge",
        sagemaker_session=SAGEMAKER_SESSION
    ),
    inputs=[ProcessingInput(
        source=f"s3://{s3_bucket_param}/{s3_folder_param}",
        destination="/opt/ml/processing/input"
    )],
    outputs=[
        ProcessingOutput(output_name="train_data", source="/opt/ml/processing/train", destination=f"s3://{s3_bucket_param}/{s3_folder_param}/train_subset"),
        ProcessingOutput(output_name="val_data", source="/opt/ml/processing/val", destination=f"s3://{s3_bucket_param}/{s3_folder_param}/val_subset"),
        ProcessingOutput(output_name="test_data", source="/opt/ml/processing/test", destination=f"s3://{s3_bucket_param}/{s3_folder_param}/test_subset")
    ],
    code="pre-process.py",
    source_dir="scripts"
)

# Training + evaluation + ONNX export
train_eval_export_step = ProcessingStep(
    name="YOLO-TrainEvaluateExport",
    processor=ScriptProcessor(
        image_uri=PROCESSING_IMAGE,
        command=["python3"],
        role=ROLE,
        instance_count=1,
        instance_type="ml.m5.2xlarge",
        sagemaker_session=SAGEMAKER_SESSION
    ),
    inputs=[
        ProcessingInput(source=preprocess_step.properties.ProcessingOutputConfig.Outputs["train_data"].S3Output.S3Uri, destination="/opt/ml/processing/train"),
        ProcessingInput(source=preprocess_step.properties.ProcessingOutputConfig.Outputs["val_data"].S3Output.S3Uri, destination="/opt/ml/processing/val"),
        ProcessingInput(source=preprocess_step.properties.ProcessingOutputConfig.Outputs["test_data"].S3Output.S3Uri, destination="/opt/ml/processing/test")
    ],
    outputs=[
        ProcessingOutput(output_name="model_output", source="/opt/ml/model", destination=f"s3://{s3_bucket_param}/{s3_folder_param}/model"),
        ProcessingOutput(output_name="onnx_output", source="/opt/ml/processing/onnx_model", destination=f"s3://{s3_bucket_param}/{s3_folder_param}/onnx_models"),
        ProcessingOutput(output_name="evaluation_output", source="/opt/ml/processing/evaluation", destination=f"s3://{s3_bucket_param}/{s3_folder_param}/evaluation")
    ],
    code="train_evaluate_export.py",
    source_dir="scripts",
    job_arguments=[
        "--model", model_param,
        "--epochs", str(epochs_param.default_value),
        "--batch", str(batch_size_param.default_value)
    ]
)

# Deployment step
deploy_step = ProcessingStep(
    name="YOLO-Deploy",
    processor=ScriptProcessor(
        image_uri=PROCESSING_IMAGE,
        command=["python3"],
        role=ROLE,
        instance_count=1,
        instance_type="ml.m5.large",
        sagemaker_session=SAGEMAKER_SESSION
    ),
    inputs=[
        ProcessingInput(source=train_eval_export_step.properties.ProcessingOutputConfig.Outputs["onnx_output"].S3Output.S3Uri, destination="/opt/ml/processing/model")
    ],
    code="deploy.py",
    source_dir="scripts"
)

# Build pipeline
pipeline = Pipeline(
    name="YOLO-Object-Detection-Pipeline",
    parameters=[s3_bucket_param, s3_folder_param, model_param, epochs_param, batch_size_param],
    steps=[preprocess_step, train_eval_export_step, deploy_step]
)

pipeline.upsert(role_arn=ROLE)
logging.info("Pipeline definition uploaded successfully.")
