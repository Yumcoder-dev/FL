# MRI Brain Tumor Detection with Federated Learning

This project demonstrates a practical implementation of Federated Learning (FL) for MRI brain tumor detection using NVIDIA FLARE. The goal is to build a robust model for detecting brain tumors from MRI scans while preserving patient privacy. By using Federated Learning, multiple institutions can collaborate without sharing sensitive data, leveraging distributed datasets for improved model accuracy and generalization.

## Objectives

1. **Develop a robust MRI brain tumor detection model** using Federated Learning to ensure data privacy.
2. **Utilize distributed datasets** across multiple institutions to improve model accuracy and generalization.

## Datasets

1. **Primary Dataset**: 
   - **Kaggle Brain Tumor Classification (MRI)**: This dataset provides labeled MRI scans for brain tumor detection.
2. **Additional Neuroimaging Data**:
   - **OpenNeuro**: A source for additional fMRI and MRI datasets, following the Brain Imaging Data Structure (BIDS) format, which provides neuroimaging data for improved model performance and validation.

## Background Research

The project builds upon recent advancements in FL applied to medical imaging. 

Key aspects of FL in medical imaging:
- **Privacy**: FL allows hospitals and research institutions to collaborate without sharing raw data, maintaining patient confidentiality.
- **Generalization**: By leveraging data from different sources, the model can better generalize across various MRI scanners, protocols, and populations.
- **Compliance**: FL helps meet regulatory requirements, as sensitive data never leaves the originating institution.

### Implementation Method

1. **Install NVIDIA FLARE**:
   - Follow the NVIDIA FLARE [installation guide](https://github.com/NVIDIA/NVFlare) to set up the environment.

2. **Define the Job Configuration**:
here - The `fedavg_script_runner_hello_numpy.py` script provides a base example of defining the FL job with FedAvg aggregation.
   - Update the script to configure parameters for your specific data and model:
     - `n_clients`: Number of clients participating.
     - `num_rounds`: Number of training rounds.
     - `train_script`: Path to the MRI training script on each client.

3. **Implement Data Preprocessing**:
   - Each client pre-processes MRI data, converting it to the BIDS format where applicable, to maintain consistent data structure.

4. **Train Local Models**:
   - Each institution trains its local model on MRI scans.
   - Use NVIDIA FLARE's `ScriptRunner` to manage the training script execution.

5. **Federated Aggregation**:
   - After each training round, model weights are sent to the central server, where the FedAvg algorithm aggregates them.
   - The aggregated model is then distributed back to each client.

6. **Validation**:
   - Evaluate model performance on each client’s local validation set.

## Example Configuration

Below is an example configuration of the FedAvg workflow using the script `fedavg_script_runner_hello_numpy.py`:

```python
from nvflare import FedJob
from nvflare.app_common.workflows.fedavg import FedAvg
from nvflare.job_config.script_runner import FrameworkType, ScriptRunner

n_clients = 2  # Number of participating clients
num_rounds = 3  # Number of FL rounds
train_script = "client/hello-numpy_fl.py"  # Path to the client training script

job = FedJob(name="hello-fedavg-numpy")
controller = FedAvg(
    num_clients=n_clients,
    num_rounds=num_rounds,
)

for i in range(n_clients):
    executor = ScriptRunner(script=train_script, framework=FrameworkType.NUMPY)
    job.to(executor, f"site-{i+1}")

job.simulator_run("workdir", gpu="0")
```

## Additional Resources

- **Brain Imaging Data Structure (BIDS)**: Ensure all MRI data follows the BIDS format for standardization. More info [here](https://bids.neuroimaging.io/).
- **Federated Learning with NVIDIA FLARE**: Comprehensive documentation can be found on the [NVIDIA FLARE GitHub page](https://github.com/NVIDIA/NVFlare).
