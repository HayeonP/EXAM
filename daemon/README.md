# EXAM Daemon

ResNet18 example daemon with TensorRT GPU, PyTorch, and mixed-worker execution.

## Scripts

| Script | Action |
| --- | --- |
| `01_run_resnet18_three_process_fifo_test.sh` | Run three TensorRT clients in FIFO order. |
| `02_run_resnet18_three_process_interleave_test.sh` | Run three TensorRT clients with interleaved SG execution. |
| `03_run_resnet18_pytorch_tensorrt_interleave_test.sh` | Interleave one PyTorch client and one TensorRT client. |
| `04_compare_pytorch_direct_vs_worker_response_time.sh` | Compare direct PyTorch/LibTorch with PyTorch worker latency. |
| `05_run_resnet18_pytorch_tensorrt_mixed.sh` | Classify sample images with PyTorch, TensorRT, and mixed SGs, and measure their response times. |
| `terminate.sh` | Stop EXAM processes and remove EXAM SHM files. |

## Artifacts

| Path | Content |
| --- | --- |
| `artifacts/image/` | Three PNG inputs and ImageNet labels. |
| `artifacts/resnet18/` | Pretrained ResNet18 ONNX, TorchScript, and FP32 TensorRT plans. |

## Config

| Key | Meaning |
| --- | --- |
| `config/exam.yaml` | Default daemon/runtime config. |
| `EXAM_CONFIG_PATH` | Override config path. |
| `cpu_affinity.daemon` | CPU ids for the daemon thread. |
| `cpu_affinity.tensor_rt_gpu_worker` | CPU ids for the TensorRT GPU worker. |
| `cpu_affinity.pytorch_worker` | CPU ids for the PyTorch worker and direct PyTorch comparisons. |

Config lookup order is `EXAM_CONFIG_PATH`, `config/examl.yaml`, then `config/exam.yaml`.

## Apps

| App | Action |
| --- | --- |
| `exam_daemon_example.cpp` | Starts `exam_daemon`. |
| `resnet18_process.cpp` | Sends image/tensor inputs through EXAM. |
| `resnet18_pytorch_direct.cpp` | Runs split TorchScript directly with LibTorch. |
| `resnet18_pytorch_direct.py` | Runs split TorchScript directly with Python PyTorch. |
| `shared_memory_region_sample.cpp` | Small shared-memory API sample. |
