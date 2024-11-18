# Fast Canny

Fast and furious Canny edge detection.

## Pre-requisites

- Cmake >= 3.26
- GCC >= 8.5.0

## Build

```bash

cmake -B build -G "Unix Makefiles"
cmake --build build

```
NOTE: When building the whole project, especifally if you are building the opencv benchmark, CMake will download the opencv source code and build it. This will take a while.

## Development

This project is using Devcontainers to provide a consistent development environment. To start the development container, you will need to have Docker installed on your machine. And then on VSCode you can open the project in a container by clicking on the green button on the bottom left corner of the window and selecting "Reopen in Container".


## Run

### Running OpenCV benchmark

To run the OpenCV benchmark, you will need to unzip the images from the COCO dataset.

```bash
unzip coco_images.zip
```

Then you can run the benchmark with the following command:

```bash
./build/benchmark/opencv_benchmark coco_images/
```

### Running Gaussian Filter benchmark

To run the Gaussian Filter benchmark, you can run the following command:

```bash
./build/benchmark/gaussian_filter_benchmark
```


### Running the image download script

The images in the zip file are downloaded by running the python script `download_coco_images.py`. This script will download the images from the COCO dataset and save them in the `coco_images` folder.

For local python development, you should set up and run the Python virtual environment. It will set up a Python 3.12 virtualenv. When running the benchmark on the ECE machines, we don't need to redownload the images.

```bash
 # Only for local development
pipenv shell
pipenv install
python download_coco_images.py
```

### Generating assmebly code
To generate the assembly code for the Gaussian filter benchmark, you can run the following command:

```bash
objdump -d build/benchmark/gaussian_filter_benchmark > disassembly.S
```
