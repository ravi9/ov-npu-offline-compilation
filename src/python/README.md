#  OpenVINO Off-Target NPU Compilation Python Sample

This sample demonstrates how to use OpenVINO off-target compilation to compile a model for an NPU target device. The sample consists of two parts:
1. Off-target compilation on a host system (e.g., Xeon CPU) to generate a blob file for the target NPU device. See NPU device configuration details [here](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md).
2. Running the compiled blob file on the target system with the NPU.

For C++ sample, please refer to the [NPU Compile Tool](https://github.com/openvinotoolkit/openvino/tree/master/src/plugins/intel_npu/tools/compile_tool).

---

### Step 1: Setup on Off-Target System (ex. Xeon)

1. Install OpenVINO Runtime from an archive file as it includes `libopenvino_intel_npu_compiler.so` needed for off-target compilation.

- Follow the guide to install OpenVINO Runtime from an archive file: [Linux](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html) | [Windows](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-windows.html)

- **Linux:**

    <details>
    <summary>📦 Click to expand OpenVINO 2026.0 installation from an archive file on Ubuntu</summary>
    <br>

    ```bash
    wget https://raw.githubusercontent.com/ravi9/misc-scripts/main/openvino/ov-archive-install/install-openvino-from-archive.sh
    chmod +x install-openvino-from-archive.sh
    ./install-openvino-from-archive.sh
    ```

    Verify OpenVINO is initialized properly:
    ```bash
    echo $OpenVINO_DIR
    ```
    </details>

2. Create a Python virtual environment and install the required dependencies:
```bash
python3 -m venv ov-off-target-env
source ov-off-target-env/bin/activate
pip install torch torchvision requests pillow --extra-index-url https://download.pytorch.org/whl/cpu 
``` 

3. Download the sample code and run:
```bash
wget https://raw.githubusercontent.com/ravi9/ov-npu-offline-compilation/refs/heads/main/src/python/off-target-export.py

source ov-off-target-env/bin/activate
source /opt/intel/openvino/bin/setupvars.sh
python off-target-export.py
```

### Step 2: Setup on Target System with NPU

1. Install the NPU drivers on the target system. For detailed instructions, see: [Additional Configurations for Hardware Acceleration](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html) 

2. Create a Python virtual environment and install the required dependencies:

```bash
python3 -m venv ov-target-env
source ov-target-env/bin/activate
pip install torch torchvision requests pillow --extra-index-url https://download.pytorch.org/whl/cpu 
pip install openvino
``` 

3. Download the blob file from the off-target system to the target system 

4. Download the sample code and run:
```bash
wget https://raw.githubusercontent.com/ravi9/ov-npu-offline-compilation/refs/heads/main/src/python/target-test.py

source ov-target-env/bin/activate
python target-test.py
```

### Expected Output:
On the *off-target system*, you should see output similar to the following, indicating that the model was successfully compiled for the NPU and the blob file was exported:
```console
$ python off-target-export.py 
OpenVINO version: 2026.0.0-20965-c6d6a13a886-releases/2026/0
Pytorch version: 2.11.0+cpu
Available devices: ['CPU']

Testing PyTorch model on CPU...
PyTorch: tabby: 20.8%

Converting model to OpenVINO format...

Compiling and Testing OV model for CPU...
OpenVINO (CPU): tabby: 20.8%

Compiling model for NPU and exporting blob...
Blob exported to ./xeon_npu_cache/resnet50_src_xeon_target_LNL.blob
```


On the *target system* with NPU, you should see output similar to the following:
```console
$ python target-test.py 
OpenVINO version: 2026.2.0-21403-9a9b7d7ea00
Pytorch version: 2.11.0+cpu
Available devices: ['CPU', 'GPU', 'NPU']

Loading OV compiled model blob on NPU...

Testing OV model on NPU...
OpenVINO (NPU) Output: tabby: 19.5%
```

### Additional Resources
- [OpenVINO Documentation](https://docs.openvino.ai/latest/index.html)
- [OpenVINO NPU Docs](https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_npu/README.md)
