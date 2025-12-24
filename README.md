# End_to_End_AI_SageMaker

## Introduction

This setup integrates multiple AWS services (SageMaker, Lambda, API Gateway) to automate the training and deployment of an object detection model and expose it via a REST API for real-time inference. Make sure to replace all placeholders like your-sagemaker-role, your-bucket, your-processing-image-uri, your-model-image-uri, and API Gateway URLs with your actual values.

## Features

- End-to-end machine learning lifecycle with SageMaker integration
- Automated data preprocessing, model training, and evaluation
- Model deployment with managed endpoints on AWS
- Real-time and batch inference capabilities
- Monitoring and logging for deployed models
- Modular and extensible code structure for custom ML use cases

## Requirements

Before starting, ensure you have the following:

- Python 3.7 or higher
- AWS account with appropriate SageMaker permissions
- AWS CLI configured with your credentials
- SageMaker Studio or notebook instance access (optional but recommended)
- Required Python packages (see installation section)

## Installation

To set up the repository and its dependencies, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SameaSaeed/End_to_End_AI_SageMaker.git
   cd End_to_End_AI_SageMaker
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure AWS CLI if not already done:**
   ```bash
   aws configure
   ```

## Usage

The repository provides scripts and notebooks to guide you through each stage of the ML workflow. The typical process includes:

1. **Data Preparation**
   - Prepare your dataset and upload it to an S3 bucket.
   - Modify configuration files as needed for your data paths.

2. **Model Training**
   - Run the training script or notebook to launch a SageMaker training job.
   - Monitor training progress through SageMaker Console or CloudWatch.

3. **Model Evaluation**
   - Evaluate model performance using built-in metrics or custom scripts.
   - Review logs/output for insights and further tuning.

4. **Model Deployment**
   - Deploy the trained model to a SageMaker endpoint for real-time inference.
   - Use provided scripts to manage endpoint lifecycle (create/update/delete).

5. **Inference and Monitoring**
   - Send prediction requests to the endpoint via sample client scripts.
   - Monitor endpoint metrics and logs for performance and errors.

### Example Training Command

```bash
python train.py --config configs/train_config.yaml
```

### Example Deployment Command

```bash
python deploy.py --model-artifact s3://bucket/model.tar.gz --endpoint-name my-endpoint
```

### Example Inference Command

```bash
python predict.py --endpoint-name my-endpoint --input data/sample_input.json
```

### Example E2E Pipeline Command

```bash
python pipeline.py
```
### Example Invoke Endpoint Command

Create a new REST API in API Gateway:
1. In the API Gateway Console, create a new REST API.
2. Create a new resource (e.g., /object-detection).
3. Create a new POST method for the resource.
4. Set the Integration Type to Lambda Function and select the lambda_invoke_model.py function.

Deploy the API:
1. In API Gateway, click on Actions > Deploy API.
2. Select the deployment stage (e.g., prod).
3. Save and note the Invoke URL for your API.

Send a POST request to the API Gateway endpoint:

```bash
curl -X POST "https://your-api-id.execute-api.us-west-2.amazonaws.com/prod/object-detection" \
-H "Content-Type: application/json" \
-d '{"input_data": {"image": "base64-encoded-image"}}'
```

## Configuration

The repository uses YAML or JSON configuration files to specify key parameters such as:

- S3 bucket locations for datasets and model artifacts
- Training hyperparameters (batch size, epochs, learning rate)
- SageMaker instance types for training and inference
- Endpoint names and deployment options

Ensure you update the configuration files in the `configs/` directory to match your AWS setup and desired experiment settings.

## Contributing

Contributions are welcome! To contribute:

- Fork the repository and create your branch.
- Ensure code style consistency with existing codebase.
- Write clear commit messages and document your changes.
- Submit a pull request with a description of your contribution.
- For major changes, open an issue first to discuss your proposal.

---

We welcome feedback and suggestions to improve this project. Please open issues for bugs, feature requests, or general inquiries. Thank you for your interest in End_to_End_AI_SageMaker!
