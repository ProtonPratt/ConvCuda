#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <filesystem>
#include <string>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>


namespace solution {

    // Error checking macro
    #define CUDA_ERROR_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); } 
    inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
        if (code != cudaSuccess){
            fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    // CUDA kernel for 2D convolution
    __global__ void convolution2D(const float* d_img, const float* d_kernel, float* d_result, int num_rows, int num_cols) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;

        // Assuming a 3x3 kernel
        const int kernel_size = 3;
        const int half_kernel = kernel_size / 2;

        if (x >= num_cols || y >= num_rows) {
            return;  // Skip threads outside the image bounds
        }

        float sum = 0;
        for (int ky = -half_kernel; ky <= half_kernel; ++ky) {
            for (int kx = -half_kernel; kx <= half_kernel; ++kx) {
                int img_x = x + kx;
                int img_y = y + ky;
                if (img_x >= 0 && img_x < num_cols && img_y >= 0 && img_y < num_rows) {
                    sum += d_img[img_y * num_cols + img_x] * d_kernel[(ky + half_kernel) * kernel_size + (kx + half_kernel)];
                }
            }
        }

        // Store the result
        d_result[y * num_cols + x] = sum;
    }

    // Function to compute the convolution of an image using a given kernel
    std::string compute(const std::string &bitmap_path, const float kernel[3][3], const int num_rows, const int num_cols) {
        // Output file path
        std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";

        // Open the input image file
        int fd = open(bitmap_path.c_str(), O_RDONLY);
        if (fd == -1) {
            std::cerr << "Failed to open file: " << bitmap_path << std::endl;
            return "";
        }

        // Obtain file size using stat()
        struct stat file_stat;
        if (fstat(fd, &file_stat) == -1) {
            std::cerr << "Failed to stat file: " << bitmap_path << std::endl;
            close(fd);
            return "";
        }
        size_t file_size = file_stat.st_size;

        // Map the file into memory
        void* mapped_memory = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped_memory == MAP_FAILED) {
            std::cerr << "Failed to map file: " << bitmap_path << std::endl;
            close(fd);
            return "";
        }

        // Convert the mapped memory to a pointer to float
        auto img = reinterpret_cast<float*>(mapped_memory);

        // Allocate memory on the device
        float *d_img, *d_kernel, *d_result;
        CUDA_ERROR_CHECK(cudaMalloc((void**)&d_img, sizeof(float) * num_rows * num_cols));
        CUDA_ERROR_CHECK(cudaMalloc((void**)&d_kernel, sizeof(float) * 3 * 3)); // Assuming a 3x3 kernel
        CUDA_ERROR_CHECK(cudaMalloc((void**)&d_result, sizeof(float) * num_rows * num_cols));

        // Transfer data from mapped memory to device memory
        CUDA_ERROR_CHECK(cudaMemcpy(d_img, img, sizeof(float) * num_rows * num_cols, cudaMemcpyHostToDevice));
        CUDA_ERROR_CHECK(cudaMemcpy(d_kernel, kernel, sizeof(float) * 3 * 3, cudaMemcpyHostToDevice));

        // Configure grid and block size for kernel launch
        dim3 blockSize(16, 16); // Adjust as needed
        dim3 gridSize((num_cols + blockSize.x - 1) / blockSize.x, (num_rows + blockSize.y - 1) / blockSize.y);

        // Launch the CUDA kernel
        convolution2D<<<gridSize, blockSize>>>(d_img, d_kernel, d_result, num_rows, num_cols);
        CUDA_ERROR_CHECK(cudaGetLastError());

        // Transfer the result back from device to host
        auto result = std::make_unique<float[]>(num_rows * num_cols);
        CUDA_ERROR_CHECK(cudaMemcpy(result.get(), d_result, sizeof(float) * num_rows * num_cols, cudaMemcpyDeviceToHost));

        // Write the result to output file
        std::ofstream sol_fs(sol_path, std::ios::binary);
        sol_fs.write(reinterpret_cast<char*>(result.get()), sizeof(float) * num_rows * num_cols);
        sol_fs.close();

        // Unmap the file from memory
        munmap(mapped_memory, file_size);

        // Close the file descriptor
        close(fd);

        // Free device memory
        CUDA_ERROR_CHECK(cudaFree(d_img));
        CUDA_ERROR_CHECK(cudaFree(d_kernel));
        CUDA_ERROR_CHECK(cudaFree(d_result));

        return sol_path; // Return path to the output file
    }

}

// namespace solution{
//         #define CUDA_ERROR_CHECK(ans) { cudaAssert((ans), __FILE__, __LINE__); } 
//         inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true){
//                 if (code != cudaSuccess){
//                         fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
//                         if (abort) exit(code);
//                 }
//         }

//         __global__ void convolution2D(float* img, const float* kernel, float* result, int num_rows, int num_cols) {
//                 int x = blockIdx.x * blockDim.x + threadIdx.x;
//                 int y = blockIdx.y * blockDim.y + threadIdx.y;

//                 // Assuming a 3x3 kernel
//                 float kernel_size = 3;
//                 float half_kernel = kernel_size / 2.0;

//                 float sum = 0;
//                 for (int ky = -1; ky <= 1; ++ky) {
//                         for (int kx = -1; kx <= 1; ++kx) {
//                         int img_y = y + ky;
//                         int img_x = x + kx;
//                         if (img_y >= 0 && img_y < num_rows && img_x >= 0 && img_x < num_cols) {
//                                 sum += img[img_y * num_cols + img_x] * kernel[(ky + 1) * 3 + (kx + 1)];
//                         }
//                         }
//                 }

//                 if (x < num_cols && y < num_rows) {
//                         result[y * num_cols + x] = sum;
//                 }
//         }


//         std::string compute(const std::string &bitmap_path, const float kernel[3][3], const std::int32_t num_rows, const std::int32_t num_cols){
//                 std::string sol_path = std::filesystem::temp_directory_path() / "student_sol.bmp";
//                 std::ofstream sol_fs(sol_path, std::ios::binary);
//                 std::ifstream bitmap_fs(bitmap_path, std::ios::binary);
//                 const auto img = std::make_unique<float[]>(num_rows * num_cols);
//                 bitmap_fs.read(reinterpret_cast<char*>(img.get()), sizeof(float) * num_rows * num_cols);
//                 // Do some allocations etc.
//                 // Call CUDA Kernel
//                 dim3 blockSize(16, 16);
//                 dim3 gridSize((num_cols + blockSize.x - 1) / blockSize.x, (num_rows + blockSize.y - 1) / blockSize.y);

//                 convolution2D<<<gridSize, blockSize>>>(d_img, d_kernel, d_result, num_rows, num_cols);
//                 CUDA_ERROR_CHECK(cudaGetLastError());
//                 bitmap_fs.close();
//                 return sol_path;
//         }
// };
