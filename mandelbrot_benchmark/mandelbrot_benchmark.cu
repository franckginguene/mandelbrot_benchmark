/***************************************************************************
* Copyright (c) Franck Ginguene											   * 
*                                                                          *
* Bench mutiple implementations of the mandelbrot problem                  *
*  - Scalar																   *
*  - SIMD with xsimd													   *
*  - OpenMP																   *
*  - OpenMP	+ SIMD														   *
*  - CUDA	         													   *
****************************************************************************/

#include <cstdio>
#include <iostream>
#include <string>
#include <omp.h>
#include <fstream>
#include <xsimd-master/examples/pico_bench.hpp>
#include <xsimd-master/include/xsimd/xsimd.hpp>

#define XSIMD_ENABLE_FALLBACK

// helper function to write the rendered image as PPM file
inline void writePPM(const std::string &fileName,
	const int sizeX,
	const int sizeY,
	const int *pixel)
{
	FILE* file = fopen(fileName.c_str(), "wb");
	fprintf(file, "P6\n%i %i\n255\n", sizeX, sizeY);
	unsigned char* out = (unsigned char*)alloca(3 * sizeX);
	for (int y = 0; y < sizeY; y++)
	{
		const unsigned char* in =
			(const unsigned char*)&pixel[(sizeY - 1 - y) * sizeX];

		for (int x = 0; x < sizeX; x++)
		{
			out[3 * x + 0] = in[4 * x + 0];
			out[3 * x + 1] = in[4 * x + 1];
			out[3 * x + 2] = in[4 * x + 2];
		}

		fwrite(out, 3 * sizeX, sizeof(char), file);
	}
	fprintf(file, "\n");
	fclose(file);
}

namespace cuda {
	__global__ void mandel(int *out, int width, int height, int maxiter, float x0, float y0, float dx, float dy)
	{
		unsigned int x_dim = blockIdx.x*blockDim.x + threadIdx.x;
		unsigned int y_dim = blockIdx.y*blockDim.y + threadIdx.y;
		int index = width*y_dim + x_dim;
		float x_origin(x0 + x_dim * dx);
		float y_origin(y0 + y_dim * dy);

		float x = 0.0;
		float y = 0.0;

		int iteration = 0;
		while (x*x + y * y <= 4 && iteration < maxiter) {
			float xtemp = x * x - y * y + x_origin;
			y = 2 * x*y + y_origin;
			x = xtemp;
			iteration++;
		}

		if (iteration == maxiter) {
			out[index] = 0;
		}
		else {
			out[index] = iteration;
		}
	}

	void mandelbrot(float x0, float y0, float x1, float y1,
		int width, int height, int maxIterations, int output[])
	{
		// Multiply by 3 here, since we need red, green and blue for each pixel
		size_t buffer_size = sizeof(int) * width * height ;
		
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;

		int *image;
		cudaMalloc((void **)&image, buffer_size);


		dim3 blockDim(16, 16, 1);
		dim3 gridDim(width / blockDim.x, height / blockDim.y, 1);
		mandel <<< gridDim, blockDim, 0 >>> (image, width, height, maxIterations, x0, y0, dx, dy);

		cudaMemcpy(output, image, buffer_size, cudaMemcpyDeviceToHost);

		cudaFree(image);
	}
} // namespace xsimd

namespace xsimd {

	template <std::size_t N>
	inline batch<int, N> mandel(const batch_bool<float, N> &_active,
		const batch<float, N> &c_re,
		const batch<float, N> &c_im,
		int maxIters)
	{
		batch<float, N> z_re = c_re;
		batch<float, N> z_im = c_im;
		batch<int, N> vi(0);
		for (int i = 0; i < maxIters; ++i)
		{
			auto active = _active & ((z_re * z_re + z_im * z_im) <= batch<float, N>(4.f));
			if (!xsimd::any(active))
			{
				break;
			}

			batch<float, N> new_re = z_re * z_re - z_im * z_im;
			batch<float, N> new_im = 2.f * z_re * z_im;

			z_re = c_re + new_re;
			z_im = c_im + new_im;

			vi = select(bool_cast(active), vi + 1, vi);
		}

		return vi;
	}

	template <std::size_t N>
	void mandelbrot(float x0,
		float y0,
		float x1,
		float y1,
		int width,
		int height,
		int maxIters,
		int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;

		float arange[N];
		std::iota(&arange[0], &arange[N], 0.f);
		batch<float, N> programIndex(&arange[0], xsimd::aligned_mode());
		// std::iota(programIndex.begin(), programIndex.end(), 0.f
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i += N)
			{
				batch<float, N> x(x0 + (i + programIndex) * dx);
				batch<float, N> y(y0 + j * dy);

				auto active = x < batch<float, N>(width);

				int base_index = (j * width + i);
				auto result = mandel(active, x, y, maxIters);

				// implement masked store!
				// xsimd::store_aligned(result, output + base_index, active);
				batch<int, N> prev_data(output + base_index);
				select(bool_cast(active), result, prev_data)
					.store_aligned(output + base_index);
			}
		}
	}

} // namespace xsimd

namespace xsimdomp {

	template <std::size_t N>
	inline xsimd::batch<int, N> mandel(const xsimd::batch_bool<float, N> &_active,
		const xsimd::batch<float, N> &c_re,
		const xsimd::batch<float, N> &c_im,
		int maxIters)
	{
		xsimd::batch<float, N> z_re = c_re;
		xsimd::batch<float, N> z_im = c_im;
		xsimd::batch<int, N> vi(0);
		for (int i = 0; i < maxIters; ++i)
		{
			auto active = _active & ((z_re * z_re + z_im * z_im) <= xsimd::batch<float, N>(4.f));
			if (!xsimd::any(active))
			{
				break;
			}

			xsimd::batch<float, N> new_re = z_re * z_re - z_im * z_im;
			xsimd::batch<float, N> new_im = 2.f * z_re * z_im;

			z_re = c_re + new_re;
			z_im = c_im + new_im;

			vi = select(bool_cast(active), vi + 1, vi);
		}

		return vi;
	}

	template <std::size_t N>
	void mandelbrot(float x0,
		float y0,
		float x1,
		float y1,
		int width,
		int height,
		int maxIters,
		int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;

		float arange[N];
		std::iota(&arange[0], &arange[N], 0.f);
		xsimd::batch<float, N> programIndex(&arange[0], xsimd::aligned_mode());
		// std::iota(programIndex.begin(), programIndex.end(), 0.f
		#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; i += N)
			{
				xsimd::batch<float, N> x(x0 + (i + programIndex) * dx);
				xsimd::batch<float, N> y(y0 + j * dy);

				auto active = x < xsimd::batch<float, N>(width);

				int base_index = (j * width + i);
				auto result = xsimdomp::mandel(active, x, y, maxIters);

				// implement masked store!
				// xsimd::store_aligned(result, output + base_index, active);
				xsimd::batch<int, N> prev_data(output + base_index);
				select(bool_cast(active), result, prev_data)
					.store_aligned(output + base_index);
			}
		}
	}

} // namespace xsimdomp

// omp version ////////////////////////////////////////////////////////////////

namespace omp {

	//#pragma omp declare simd
	template <typename T>
	inline int mandel(T c_re, T c_im, int count)
	{
		T z_re = c_re, z_im = c_im;
		int i;
		for (i = 0; i < count; ++i)
		{
			if (z_re * z_re + z_im * z_im > 4.f)
			{
				break;
			}

			T new_re = z_re * z_re - z_im * z_im;
			T new_im = 2.f * z_re * z_im;
			z_re = c_re + new_re;
			z_im = c_im + new_im;
		}

		return i;
	}

	void mandelbrot(float x0, float y0, float x1, float y1, int width,
		int height, int maxIterations, int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;
		#pragma omp parallel for
		for (int j = 0; j < height; j++)
		{

			for (int i = 0; i < width; ++i)
			{
				float x = x0 + i * dx;
				float y = y0 + j * dy;

				int index = (j * width + i);
				output[index] = mandel<float>(x, y, maxIterations);
			}
		}
	}

} // namespace omp

// scalar version /////////////////////////////////////////////////////////////

namespace scalar {

	inline int mandel(float c_re, float c_im, int count)
	{
		float z_re = c_re, z_im = c_im;
		int i;
		for (i = 0; i < count; ++i)
		{
			if (z_re * z_re + z_im * z_im > 4.f)
			{
				break;
			}

			float new_re = z_re * z_re - z_im * z_im;
			float new_im = 2.f * z_re * z_im;
			z_re = c_re + new_re;
			z_im = c_im + new_im;
		}

		return i;
	}

	void mandelbrot(float x0, float y0, float x1, float y1,
		int width, int height, int maxIterations, int output[])
	{
		float dx = (x1 - x0) / width;
		float dy = (y1 - y0) / height;

		for (int j = 0; j < height; j++)
		{
			for (int i = 0; i < width; ++i)
			{
				float x = x0 + i * dx;
				float y = y0 + j * dy;

				int index = (j * width + i);
				output[index] = mandel(x, y, maxIterations);
			}
		}
	}

}  // namespace scalar

int main()
{
	using namespace std::chrono;

	omp_set_num_threads(12);
	const unsigned int width = 450;
	const unsigned int height = 450;
	const float zr = -0.743639266077433;
	const float zi = +0.131824786875559;

	const int maxIters = 256;
	constexpr size_t nbiter = 1;
	constexpr int nbpoints = 20;
	int istep = 1;
	std::vector<__int64> times_ms(nbpoints);
	std::vector<float> x0_vec(nbpoints), x1_vec(nbpoints), y0_vec(nbpoints), y1_vec(nbpoints);
	for (int i = 0; i < nbpoints; i++)
	{
		float scale = 4. * std::pow(2, -std::min(i / 60., 53.)*0.7);
		x0_vec[i] = zr - scale;
		x1_vec[i] = zr + scale;
		y0_vec[i] = zi - scale;
		y1_vec[i] = zi + scale;
	}
	alignas(64) std::array<int, width * height> buf;

	auto bencher = pico_bench::Benchmarker<microseconds>{ nbiter, seconds{10} };

	std::cout << "starting benchmarks (results in 'ms')... " << '\n';

	// export CVS
	std::ofstream times_ms_file;
	times_ms_file.open("C:/Users/Franck/Documents/projets/tedics_hpc/example.csv");

	//// scalar run ///////////////////////////////////////////////////////////////
	for (int i = 0; i < nbpoints; i += istep)
	{
		std::fill(buf.begin(), buf.end(), 0);

		auto stats_scalar = bencher([&]() {
			scalar::mandelbrot(x0_vec[i], y0_vec[i], x1_vec[i], y1_vec[i], width, height, maxIters, buf.data());
		});
		times_ms_file << stats_scalar.min().count() << ",";
		std::cout << '\n' << "scalar " << stats_scalar << '\n';
	}
	times_ms_file << "\n";
	//std::string name = "mandelbrot_scalar_" + std::to_string(i) + ".ppm";
	//writePPM(name, width, height, buf.data());
	 //   // xsimd_4 run //////////////////////////////////////////////////////////////

	for (int i = 0; i < nbpoints; i += istep)
	{
		std::fill(buf.begin(), buf.end(), 0);

		auto stats_4 = bencher([&]() {
			xsimd::mandelbrot<4>(x0_vec[i], y0_vec[i], x1_vec[i], y1_vec[i], width, height, maxIters, buf.data());
		});


		times_ms_file << stats_4.min().count() << ",";
		std::cout << '\n' << "stats_4 " << stats_4 << '\n';
	}
	times_ms_file << "\n";

	//   // omp run //////////////////////////////////////////////////////////////////
	for (int i = 0; i < nbpoints; i += istep)
	{
		std::fill(buf.begin(), buf.end(), 0);

		auto stats_omp = bencher([&]() {
			omp::mandelbrot(x0_vec[i], y0_vec[i], x1_vec[i], y1_vec[i], width, height, maxIters, buf.data());
		});

		times_ms_file << stats_omp.min().count() << ",";
		std::cout << '\n' << "omp " << stats_omp << '\n';
	}
	times_ms_file << "\n";

	//// xsimd_4_omp run //////////////////////////////////////////////////////////////

	for (int i = 0; i < nbpoints; i += istep)
	{
		std::fill(buf.begin(), buf.end(), 0);

		auto stats_4omp = bencher([&]() {
			xsimdomp::mandelbrot<4>(x0_vec[i], y0_vec[i], x1_vec[i], y1_vec[i], width, height, maxIters, buf.data());
		});


		times_ms_file << stats_4omp.min().count() << ",";
		std::cout << '\n' << "stats_4omp " << stats_4omp << '\n';
	}
	times_ms_file << "\n";

	//// cuda run //////////////////////////////////////////////////////////////

	for (int i = 0; i < nbpoints; i += istep)
	{
		std::fill(buf.begin(), buf.end(), 0);

		auto stats_cuda = bencher([&]() {
			cuda::mandelbrot(x0_vec[i], y0_vec[i], x1_vec[i], y1_vec[i], width, height, maxIters, buf.data());
		});


		times_ms_file << stats_cuda.min().count() << ",";
		std::cout << '\n' << "stats_cuda " << stats_cuda << '\n';
	}
	times_ms_file << "\n";

	times_ms_file.close();
	getchar();
	return 0;
}