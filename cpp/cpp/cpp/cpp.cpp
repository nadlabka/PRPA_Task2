#include <iostream>
#include <vector>
#include <numeric>
#include <cmath>
#include <chrono>
#include <omp.h>


// Функція для вимірювання часу виконання в секундах
float getElapsedTime(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
    return std::chrono::duration<float>(end - start).count();
}

// Функція для підрахунку суми масиву без багатопотоковості
void sumArray(float* arrA, float* arrB, float* arrC, size_t size) {
    for (size_t i = 0; i < size; i++) 
    {
        arrC[i] = arrA[i] + arrB[i];
    }
}

// Функція для підрахунку суми масиву з використанням багатопотоковості (OpenMP)
void threadedSumArray(float* arrA, float* arrB, float* arrC, size_t size) {
    auto res = omp_get_max_threads();
    #pragma omp parallel for num_threads(8)
    for (int i = 0; i < size; i++) 
    {
        arrC[i] = arrA[i] + arrB[i];
    }
}

int main() {
    size_t n = 200000000;

    float* h_a1;
    float* h_b1;
    float* h_a2;
    float* h_b2;
    float* h_c1;
    float* h_c2;

    size_t bytes = n * sizeof(float);

    h_a1 = (float*)malloc(bytes);
    h_b1 = (float*)malloc(bytes);
    h_c1 = (float*)malloc(bytes);
    h_a2 = (float*)malloc(bytes);
    h_b2 = (float*)malloc(bytes);
    h_c2 = (float*)malloc(bytes);

    for (size_t i = 0; i < n / 2; i++) {
        h_a1[2 * i] = sinf(i) * sinf(i);
        h_b1[2 * i] = cosf(i) * cosf(i);
        h_a1[2 * i + 1] = -sinf(i) * sinf(i);
        h_b1[2 * i + 1] = -cosf(i) * cosf(i);

        h_a2[2 * i] = sinf(i) * sinf(i);
        h_b2[2 * i] = cosf(i) * cosf(i);
        h_a2[2 * i + 1] = -sinf(i) * sinf(i);
        h_b2[2 * i + 1] = -cosf(i) * cosf(i);
    }


    // Без багатопотоковості
    auto startSingleThreaded = std::chrono::high_resolution_clock::now();
    sumArray(h_a1, h_b1, h_c1, n);
    auto endSingleThreaded = std::chrono::high_resolution_clock::now();

    float sum = 0;
    for (size_t i = 0; i < n; i++)
    {
        sum += h_c1[i];
    }

    // Виведення результатів та часу виконання
    std::cout << "Sequentil sum: " << sum << std::endl;
    float elapsedTimeSingleThreaded = getElapsedTime(startSingleThreaded, endSingleThreaded);
    std::cout << "Time for sequential sum: " << elapsedTimeSingleThreaded << std::endl;

    // З багатопотоковістю (OpenMP)
    auto startMultithreaded = std::chrono::high_resolution_clock::now();
    threadedSumArray(h_a2, h_b2, h_c2, n);
    auto endMultithreaded = std::chrono::high_resolution_clock::now();

    sum = 0;
    for (size_t i = 0; i < n; i++)
    {
        sum += h_c2[i];
    }

    // Виведення результатів та часу виконання
    std::cout << "Multithreaded sum: " << sum << std::endl;
    float elapsedTimeMultithreaded = getElapsedTime(startMultithreaded, endMultithreaded);
    std::cout << "Time for multithreaded sum: " << elapsedTimeMultithreaded << std::endl;

    free(h_a1);
    free(h_b1);
    free(h_c1);
    free(h_a2);
    free(h_b2);
    free(h_c2);

    return 0;
}