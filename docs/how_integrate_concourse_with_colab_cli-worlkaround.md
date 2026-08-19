# Integrating Concourse CI with Google Colab CLI

This guide outlines how to configure a **Concourse CI** pipeline to automatically execute Machine Learning workflows on **Google Colab's Cloud GPUs (A100/T4)** using the official Google Colab CLI (`colab`). This setup streams local data to the cloud, executes training scripts, exports reports to Google Drive, and logs metrics to an MLflow tracking server.

---

## Prerequisites

1. **Google Colab Account**: A paid subscription (Colab Pro/Pro+) to access premium GPUs (A100, L4, or T4).
2. **Colab Auth Token**: An authentication token generated from your Google Cloud/Colab profile.
3. **Local Dataset**: A structured data directory containing your `train/`, `val/`, and `test/` folders.
4. **MLflow Server**: A reachable MLflow instance to collect experiment parameters and metrics.

---

## 1. Concourse Pipeline Configuration (`pipeline.yml`)

This configuration uses Concourse variables (`(( ))`) to dynamically pull parameters and secrets from your environment during deployment.

```yaml
resources:
- name: source-code
  type: git
  source:
    uri: https://github.com/your-username/your-ml-repo.git
    branch: main

jobs:
- name: run-ml-gpu-experiment
  plan:
  - get: source-code
    trigger: true
  - task: upload-and-run-on-gpu
    config:
      platform: linux
      image_resource:
        type: registry-image
        source: { repository: python, tag: "3.12-slim" }
      inputs:
      - name: source-code
      params:
        LOCAL_DATA_DIR: ((local_data_dir))
        GPU_TYPE: ((gpu_type))
        COLAB_TOKEN: ((colab_auth_token))
        MLFLOW_TRACKING_URI: ((mlflow_tracking_uri))
      run:
        path: source-code/ci/run_gpu_task.sh
```

---
```yaml
platform: linux

image_resource:
  type: docker-image
  source: 
    repository: python
    tag: "3.12"

params:
  # Pull the secret JSON string into an environment variable
  COLAB_TOKEN_DATA: ((colab_token_json))

run:
  path: sh
  args:
    - -cx
    - |
      # 1. Install the CLI tool
      pip install google-colab-cli
      
      # 2. Re-create the token directory structure inside the CI container
      mkdir -p ~/.config/colab-cli
      
      # 3. Write the secret environment variable content back into the token file
      echo "$COLAB_TOKEN_DATA" > ~/.config/colab-cli/token.json
      
      # 4. Execute your remote training script safely on the target GPU
      colab exec --gpu A100 --script my_market_ml_script.py

```

## 2. Automation Shell Script (`ci/run_gpu_task.sh`)

This script runs inside the Concourse worker. It packages your data locally based on your environment variable path, provisions the cloud GPU, uploads the assets, executes the training code, and terminates the session cleanly to save your paid credits.

```bash
#!/bin/bash
set -e

# 1. Install Google Colab CLI
pip install google-colab-cli

# 2. Package your dataset from your local path
echo "Bundling dataset from: $LOCAL_DATA_DIR"
cd "$LOCAL_DATA_DIR"
tar -czf /tmp/dataset.tar.gz train/ val/ test/
cd -

# 3. Authenticate and provision the specific GPU dynamically
echo "Provisioning Google Colab Cloud GPU: $GPU_TYPE"
colab auth --token "$COLAB_TOKEN"
colab new --gpu "$GPU_TYPE" --session-name "Concourse-Run"

# 4. Stream packaged data to the cloud workspace
colab upload /tmp/dataset.tar.gz

# 5. Remote execute your Python ML script
colab exec -f source-code/src/train.py

# 6. Always shut down the instance to protect paid credits
colab stop "Concourse-Run"
```

---

## 3. Training Script Execution (`src/train.py`)

This Python script executes completely on the remote Google Colab GPU node. It unpacks the streamed dataset, handles training, pushes logs over the internet to MLflow, and drops the final summary file directly into your mounted Google Drive filesystem.

```python
import os
import tarfile
import mlflow

# 1. Extract the datasets arriving from your Concourse machine
print("Extracting training, validation, and testing assets...")
if os.path.exists("dataset.tar.gz"):
    with tarfile.open("dataset.tar.gz", "r:gz") as tar:
        tar.extractall(path="/content/data")

TRAIN_DIR = "/content/data/train"
VAL_DIR = "/content/data/val"
TEST_DIR = "/content/data/test"

# 2. Connect to your centralized MLflow tracking instance
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Paid_Colab_GPU_Pipeline")

with mlflow.start_run():
    print("Initiating cloud GPU model training pipeline...")
    # Add your PyTorch / TensorFlow training architecture here
    # ...
    
    final_loss = 0.02
    final_accuracy = 0.97
    
    mlflow.log_metric("accuracy", final_accuracy)
    mlflow.log_metric("loss", final_loss)

# 3. Mount Google Drive and dump the final evaluated experiment report
from google.colab import drive
drive.mount('/content/drive')

report_output = "/content/drive/MyDrive/YourProject/reports/run_summary.txt"
with open(report_output, "w") as f:
    f.write(f"GPU Execution Finished. Final Accuracy: {final_accuracy}\n")
print(f"Report cleanly exported straight to Google Drive: {report_output}")
```

---

## 4. Local Secrets Setup (`.env`)

Save your long-lived secrets in a local file. To feed these credentials safely into the Fly CLI, format your `.env` file using standard YAML layout syntax:

```yaml
colab_auth_token: "your-secret-google-colab-token"
mlflow_tracking_uri: "http://your-public-mlflow-server:5000"
```

---

## 5. Deploying and Switching GPUs

Deploy your pipeline using the Fly CLI. You can override the `gpu_type` and `local_data_dir` values on the fly without updating any code base repository configuration.

### For Heavy Tasks (A100 GPU):
```bash
fly -t your-concourse-target set-pipeline \
  -p ml-training-pipeline \
  -c pipeline.yml \
  --var local_data_dir="/absolute/path/to/your/project_data" \
  --var gpu_type="A100" \
  --load-vars-from=.env
```

### For Light Tasks (T4 GPU):
```bash
fly -t your-concourse-target set-pipeline \
  -p ml-training-pipeline \
  -c pipeline.yml \
  --var local_data_dir="/absolute/path/to/your/project_data" \
  --var gpu_type="T4" \
  --load-vars-from=.env
```
