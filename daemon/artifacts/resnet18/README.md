# ResNet18 Artifacts

Pretrained ImageNet ResNet18 artifacts generated from torchvision.

- `sg_sequence_tensor_rt_gpu.json`: TensorRT GPU split sequence.
- `sg_sequence_pytorch.json`: PyTorch worker split sequence.
- `sg_sequence_mixed.json`: PyTorch/TensorRT mixed split sequence.
- `onnx/`: full and split ONNX files.
- `pytorch/`: TorchScript files loaded directly by the PyTorch worker.
- `tensor_rt_gpu/`: FP32 TensorRT plans compiled with `trtexec --noTF32`.
