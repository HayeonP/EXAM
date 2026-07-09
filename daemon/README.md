# TensorRT GPU Worker

## Scripts

| Script | Summary |
| --- | --- |
| `01_run_resnet18_three_process_fifo_test.sh` | ResNet18 three-client FIFO scheduling test. |
| `02_run_resnet18_three_process_interleave_test.sh` | ResNet18 three-client interleaving scheduling test. |
| `03_run_resnet18_pytorch_tensorrt_interleave_test.sh` | PyTorch client & TensorRT GPU client interleaving test. |
| `04_compare_resnet18_response_time.sh` | PyTorch vs LibTorch vs EXAM latency test. |
| `terminate.sh` | Stop EXAM processes and SHM. |

## Optional Backends

| Backend | Summary |
| --- | --- |
| `pytorch` | Optional LibTorch worker. |

## Apps

| App | Summary |
| --- | --- |
| `exam_daemon_main.cpp` | Daemon entrypoint. |
| `process.cpp` | Generic sample client. |
| `resnet18_process.cpp` | ResNet18 hash client. |
| `resnet18_pytorch_direct.cpp` | Direct LibTorch runner. |
| `resnet18_pytorch_direct.py` | Direct Python runner. |
| `build_sample_model.cpp` | Sample artifact builder. |
| `shared_memory_region_sample.cpp` | SHM API sample. |
