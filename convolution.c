#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <unistd.h>
#else
#include <CL/cl.h>
#endif

// Function prototype for output_device_info
cl_int output_device_info(cl_device_id device);

// Function prototype for checkError
void checkError(cl_int err, const char *message);

int main() {
    cl_int err;               // error code returned from OpenCL calls
    cl_device_id device;      // compute device id 
    cl_context context;       // compute context
    cl_command_queue commands;// compute command queue
    cl_program program;       // compute program
    cl_kernel kernel;         // compute kernel
    cl_mem inputBuffer;       // input buffer
    cl_mem outputBuffer;      // output buffer
    cl_uint numPlatforms;     // number of OpenCL platforms
    int imageWidth = 1024;    // Example value, replace with actual width
    int imageHeight = 768;    // Example value, replace with actual height
    float *inputImage;        // placeholder for input image data

    // Allocate memory for input image
    inputImage = (float *)malloc(sizeof(float) * imageWidth * imageHeight);
    if (inputImage == NULL) {
        printf("Error: Failed to allocate memory for input image\n");
        return EXIT_FAILURE;
    }

    // CPU Time Measurement
    clock_t cpu_start, cpu_end;
    double cpu_time_used;

    cpu_start = clock();

    // Perform CPU operations here

    cpu_end = clock();
    cpu_time_used = ((double) (cpu_end - cpu_start)) / CLOCKS_PER_SEC;
    printf("CPU Time: %f seconds\n", cpu_time_used);

    // Create a compute context
    context = clCreateContext(0, 1, &device, NULL, NULL, &err);
    checkError(err, "Creating context");

    // Create a command queue
    commands = clCreateCommandQueue(context, device, 0, &err);
    checkError(err, "Creating command queue");

    // Find number of platforms
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    checkError(err, "Finding platforms");

    if (numPlatforms == 0) {
        printf("Error: Found 0 platforms!\n");
        return EXIT_FAILURE;
    }

    // Get all platforms
    cl_platform_id Platform[numPlatforms];
    err = clGetPlatformIDs(numPlatforms, Platform, NULL);
    checkError(err, "Getting platforms");

    // Secure a GPU
    for (int i = 0; i < numPlatforms; i++) {
        err = clGetDeviceIDs(Platform[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (err == CL_SUCCESS) {
            break;
        }
    }

    if (err != CL_SUCCESS) {
        checkError(err, "Finding a device");
    }

    err = output_device_info(device);
    checkError(err, "Printing device output");

    // GPU Time Measurement
    cl_event event;
    cl_ulong time_start, time_end;
    double total_time;

    // Create buffer for input image
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(float) * imageWidth * imageHeight, inputImage, &err);
    checkError(err, "Creating input buffer");

    // Create buffer for output image
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                   sizeof(float) * imageWidth * imageHeight, NULL, &err);
    checkError(err, "Creating output buffer");

    // Define convolution kernel source code
    const char *ConvolutionKernelSource =
        "__kernel void convolution(__global float* input, __global float* output, int width, int height) \n"
        "{ \n"
        "    int x = get_global_id(0); \n"
        "    int y = get_global_id(1); \n"
        "    float filter[3][3] = { { 0.1, 0.1, 0.1 }, { 0.1, 0.2, 0.1 }, { 0.1, 0.1, 0.1 } }; \n"
        "    float sum = 0.0; \n"
        "    for (int i = -1; i <= 1; i++) { \n"
        "        for (int j = -1; j <= 1; j++) { \n"
        "            int neighborX = clamp(x + i, 0, width - 1); \n"
        "            int neighborY = clamp(y + j, 0, height - 1); \n"
        "            sum += input[neighborY * width + neighborX] * filter[i + 1][j + 1]; \n"
        "        } \n"
        "    } \n"
        "    output[y * width + x] = sum; \n"
        "} \n";

    // Create program from kernel source
    program = clCreateProgramWithSource(context, 1, (const char **)&ConvolutionKernelSource, NULL, &err);
    checkError(err, "Creating program");

    // Build program
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    checkError(err, "Building program");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "convolution", &err);
    checkError(err, "Creating kernel");

    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuffer);
    err |= clSetKernelArg(kernel, 2, sizeof(int), &imageWidth);
    err |= clSetKernelArg(kernel, 3, sizeof(int), &imageHeight);
    checkError(err, "Setting kernel arguments");

    // Execute the kernel over the entire range of the data set
    size_t global[] = {imageWidth, imageHeight};

    // GPU Time Measurement
    err = clEnqueueNDRangeKernel(commands, kernel, 2, NULL, global, NULL, 0, NULL, &event);
    checkError(err, "Enqueueing kernel");

    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

    total_time = (double)(time_end - time_start) / 1e9; // Convert nanoseconds to seconds
    printf("GPU Time: %f seconds\n", total_time);

    // Wait for the command queue to get serviced before reading back results
    clFinish(commands);

    // Release OpenCL resources
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(outputBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    // Free allocated memory
    free(inputImage);

    return 0;
}

// Function to output device information
cl_int output_device_info(cl_device_id device) {
    cl_int err;
    char buffer[10240];
    printf("-----------------------------------------------------\n");
    // Print device name
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(buffer), buffer, NULL);
    checkError(err, "Getting device name");
    printf("Device: %s\n", buffer);
    // Print device OpenCL version
    err = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(buffer), buffer, NULL);
    checkError(err, "Getting device OpenCL version");
    printf("  OpenCL version: %s\n", buffer);
    // Print device type
    cl_device_type type;
    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(type), &type, NULL);
    checkError(err, "Getting device type");
    if (type & CL_DEVICE_TYPE_CPU)
        printf("  Type: CPU\n");
    else if (type & CL_DEVICE_TYPE_GPU)
        printf("  Type: GPU\n");
    else if (type & CL_DEVICE_TYPE_ACCELERATOR)
        printf("  Type: Accelerator\n");
    else
        printf("  Type: Unknown\n");
    printf("-----------------------------------------------------\n");
    return CL_SUCCESS;
}

// Function to check OpenCL error codes
void checkError(cl_int err, const char *message) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: %s (%d)\n", message, err);
        exit(EXIT_FAILURE);
    }
}
