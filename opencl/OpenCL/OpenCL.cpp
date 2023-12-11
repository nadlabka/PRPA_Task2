#define PROGRAM_FILE "vecAdd.cl"
#define KERNEL_FUNC "vecAdd"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <windows.h>

#ifdef MAC
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <chrono>
#include <map>
#include <iostream>

// Функція для вимірювання часу виконання в секундах
float getElapsedTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<float>(end - start).count();
}


// Функція для обчислення суми векторів без використання багатопоточності
void sumVectorsSequential(float* a, float* b, float* c, size_t n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char* argv[]) {
    // Довжина векторів
    size_t n = 200000000;

    // Вектори введення на хості
    float* h_a;
    float* h_b;
    float* h_a2;
    float* h_b2;
    // Вектор виведення на хості
    float* h_c;
    float* h_c2;

    // Буфери введення на пристрої
    cl_mem d_a;
    cl_mem d_b;

    // Буфер виведення на пристрої
    cl_mem d_c;

    cl_platform_id cpPlatform;        // Платформа OpenCL
    cl_device_id device_id;           // Ідентифікатор пристрою
    cl_context context;               // Контекст
    cl_command_queue queue;           // Черга команд
    cl_program program;               // Програма
    cl_kernel kernel;                 // Ядро

    // Розмір, в байтах, кожного вектора
    size_t bytes = n * sizeof(float);

    // Виділення пам'яті для кожного вектора на хості
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
    h_a2 = (float*)malloc(bytes);
    h_b2 = (float*)malloc(bytes);
    h_c2 = (float*)malloc(bytes);

    // Ініціалізація векторів на хості
    for (size_t i = 0; i < n / 2; i++) {
        h_a[2 * i] = sinf(i) * sinf(i);
        h_b[2 * i] = cosf(i) * cosf(i);
        h_a[2 * i + 1] = -sinf(i) * sinf(i);
        h_b[2 * i + 1] = -cosf(i) * cosf(i);

        h_a2[2 * i] = sinf(i) * sinf(i);
        h_b2[2 * i] = cosf(i) * cosf(i);
        h_a2[2 * i + 1] = -sinf(i) * sinf(i);
        h_b2[2 * i + 1] = -cosf(i) * cosf(i);
    }

    auto startSingleThreaded = std::chrono::high_resolution_clock::now();

    // Виклик функції для обчислення суми без використання багатопоточності
    sumVectorsSequential(h_a, h_b, h_c, n);

    auto endSingleThreaded = std::chrono::high_resolution_clock::now();

    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += h_c[i];
    }

    // Виведення результату та часу виконання
    printf("Sequential sum: %f\n", sum);

    float elapsedTimeSingleThreaded = getElapsedTime(startSingleThreaded, endSingleThreaded);
    printf("Time for sequential sum: %f seconds\n", elapsedTimeSingleThreaded);

    size_t globalSize, localSize;
    cl_int err;

    // Кількість робочих елементів у кожній локальній робочій групі
    localSize = 256;

    // Загальна кількість робочих елементів - localSize повинен бути дільником
    globalSize = ceil(n / (float)localSize) * localSize;

    // Прив'язка до платформи
    cl_uint num_platforms;
    clGetPlatformIDs(0, nullptr, &num_platforms);

    cl_platform_id* platforms = new cl_platform_id[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, nullptr);

    std::map<cl_uint, cl_device_id> allGPUs = {};
    for (cl_uint i = 0; i < num_platforms; ++i) {
        cl_uint num_devices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);

        if (num_devices > 0) 
        {
            cl_device_id* devices = new cl_device_id[num_devices];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices, nullptr);

            for (cl_uint j = 0; j < num_devices; ++j) 
            {
                cl_uint clock_frequency;
                clGetDeviceInfo(devices[j], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), &clock_frequency, nullptr);
                allGPUs.insert({clock_frequency, devices[j]});
            }
        }
    }
    device_id = allGPUs.rbegin()->second;

    // Створення контексту  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    // Створення черги команд 
    cl_queue_properties properties[] = { CL_QUEUE_PROPERTIES, 0, 0 };
    queue = clCreateCommandQueueWithProperties(context, device_id, properties, &err);

    const char* kernelSource =
        "__kernel void vecAdd(__global float* a, __global float* b, __global float* c, unsigned int n) { \n"
        "    size_t id = get_global_id(0); \n"
        "    id = id % n;  \n"
        "        c[id] = a[id] + b[id]; \n"
        "} \n";

    // Створення обчислювальної програми з буфера з джерела
    program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, NULL, &err);

    // Збірка виконуваного файлу програми 
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    // Створення обчислювального ядра в програмі, яку ми хочемо виконати
    kernel = clCreateKernel(program, "vecAdd", &err);

    // Створення масивів введення та виведення в пам'яті пристрою для нашого обчислення
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);

    // Запис нашого набору даних у вхідний масив у пам'яті пристрою
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
        bytes, h_a2, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
        bytes, h_b2, 0, NULL, NULL);

    // Встановлення аргументів для нашого обчислювального ядра
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);

    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
        0, NULL, NULL);

    // Очікування обслуговування черги команд перед читанням результатів
    clFinish(queue);

    auto startMultithreaded = std::chrono::high_resolution_clock::now();

    // Читання результатів з пристрою
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes, h_c2, 0, NULL, NULL);
    
    auto endMultithreaded = std::chrono::high_resolution_clock::now();

    // Виведення результату та часу виконання
    sum = 0;
    for (size_t i = 0; i < n; i++)
    {
        sum += h_c2[i];
    }
    printf("Multithreaded result: %f\n", sum);

    float elapsedTimeMultithreaded = getElapsedTime(startMultithreaded, endMultithreaded);
    printf("Time for multithreaded sum: %f seconds\n", elapsedTimeMultithreaded);

    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_a2);
    free(h_b2);
    free(h_c2);

    return 0;
};