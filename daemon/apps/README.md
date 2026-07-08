# Apps

| File | Summary |
| --- | --- |
| `exam_daemon_main.cpp` | Starts `ExamDaemon` with TensorRT GPU and mock workers, using the requested scheduling policy. |
| `process.cpp` | Generic sample client that registers an SG sequence, submits requests, waits, and prints text-oriented output. |
| `resnet18_process.cpp` | ResNet18 client that sends deterministic float32 inputs and prints input/output hashes. |
| `build_sample_model.cpp` | Generates the fake sample model artifacts, including mock files, TensorRT plans, and `sg_sequence.json`. |
| `shared_memory_region_sample.cpp` | Demonstrates shared-memory create/open/read/write/unlink behavior. |
