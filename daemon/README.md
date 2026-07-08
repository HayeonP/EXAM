# TensorRT GPU Worker

## Scripts

| Script | Summary |
| --- | --- |
| `01_run_single_process_test.sh` | Builds the project and runs one client request through the configured SG sequence. |
| `02_run_two_process_test.sh` | Runs two clients against the same SG sequence to check multi-client scheduling and cache reuse. |
| `03_run_resnet18_three_process_test.sh` | Runs three ResNet18 clients with different deterministic inputs under `mock-fifo`. |
| `04_run_resnet18_three_process_interleave_test.sh` | Forces ResNet18 SG-level interleaving and verifies interleaved output hashes match sequential baselines. |

## Apps

| App | Summary |
| --- | --- |
| `exam_daemon_main.cpp` | Starts `ExamDaemon` with TensorRT GPU and mock workers, using the requested scheduling policy. |
| `process.cpp` | Generic sample client that registers an SG sequence, submits requests, waits, and prints text-oriented output. |
| `resnet18_process.cpp` | ResNet18 client that sends deterministic float32 inputs and prints input/output hashes. |
| `build_sample_model.cpp` | Generates the fake sample model artifacts, including mock files, TensorRT plans, and `sg_sequence.json`. |
| `shared_memory_region_sample.cpp` | Demonstrates shared-memory create/open/read/write/unlink behavior. |
