/*******************************************************************
*   BoxBlur.h
*   BoxBlur
*
*	Author: Kareem Omar
*	kareem.omar@uah.edu
*	https://github.com/komrad36
*
*	Last updated Nov 3, 2016
*******************************************************************/
//
// Fastest CPU (AVX/SSE) implementation of a 128-pixel Box Blur.
//
// For even more speed see the CUDA version:
// github.com/komrad36/CUDABoxBlur
// 
// All functionality is contained in BoxBlur.h.
// 'main.cpp' is a demo and test harness.
//

#pragma once

#include <cstdint>
#include <cstdio>
#include <future>
#include <immintrin.h>
#include <random>
#include <thread>

void makeTestfile(const int width, const int height) {
	srand(35);
	FILE* fp = fopen("test.bin", "wb");
	for (int i = 0; i < width*height; ++i) {
		for (int j = 0; j < 4; ++j) {
			const auto x = static_cast<uint8_t>(rand());
			fwrite(&x, 1, 1, fp);
		}
	}
	fclose(fp);
}

void _boxBlurref(const uint8_t* const __restrict img, const int width, const int start_row, const int rows, uint8_t* const __restrict result) {
	for (int i = start_row; i < start_row + rows; ++i) {
		for (int j = 0; j < width - 128; ++j) {
			uint32_t totalR = 0U;
			uint32_t totalG = 0U;
			uint32_t totalB = 0U;
			uint32_t totalA = 0U;
			for (int k = 0; k < 128; ++k) {
				totalR += img[4*(i*width + j + k)    ];
				totalG += img[4*(i*width + j + k) + 1];
				totalB += img[4*(i*width + j + k) + 2];
				totalA += img[4*(i*width + j + k) + 3];
			}
			result[4 * (i*(width - 128) + j)    ] = static_cast<uint8_t>(totalR >> 7);
			result[4 * (i*(width - 128) + j) + 1] = static_cast<uint8_t>(totalG >> 7);
			result[4 * (i*(width - 128) + j) + 2] = static_cast<uint8_t>(totalB >> 7);
			result[4 * (i*(width - 128) + j) + 3] = static_cast<uint8_t>(totalA >> 7);
		}
	}
}

void _boxBlurScalar(const uint8_t* const __restrict img, const int width, const int start_row, const int rows, uint8_t* const __restrict result) {
	for (int i = start_row; i < start_row + rows; ++i) {
		uint32_t totalR = 0U;
		uint32_t totalG = 0U;
		uint32_t totalB = 0U;
		uint32_t totalA = 0U;
		for (int j = 0; j < 128; ++j) {
			totalR += img[4 * (i*width + j)    ];
			totalG += img[4 * (i*width + j) + 1];
			totalB += img[4 * (i*width + j) + 2];
			totalA += img[4 * (i*width + j) + 3];
		}
		result[4 * (i*(width - 128))    ] = static_cast<uint8_t>(totalR >> 7);
		result[4 * (i*(width - 128)) + 1] = static_cast<uint8_t>(totalG >> 7);
		result[4 * (i*(width - 128)) + 2] = static_cast<uint8_t>(totalB >> 7);
		result[4 * (i*(width - 128)) + 3] = static_cast<uint8_t>(totalA >> 7);
		for (int j = 1; j < width - 128; ++j) {
			totalR += img[4 * (i*width + j) + 508] - img[4 * (i*width + j) - 4];
			totalG += img[4 * (i*width + j) + 509] - img[4 * (i*width + j) - 3];
			totalB += img[4 * (i*width + j) + 510] - img[4 * (i*width + j) - 2];
			totalA += img[4 * (i*width + j) + 511] - img[4 * (i*width + j) - 1];
			result[4 * (i*(width - 128) + j)] = static_cast<uint8_t>(totalR >> 7);
			result[4 * (i*(width - 128) + j) + 1] = static_cast<uint8_t>(totalG >> 7);
			result[4 * (i*(width - 128) + j) + 2] = static_cast<uint8_t>(totalB >> 7);
			result[4 * (i*(width - 128) + j) + 3] = static_cast<uint8_t>(totalA >> 7);
		}
	}
}

template<bool single_last_column>
void processCols(const uint8_t* const __restrict img, const int width, const int i, const int j, uint8_t* const __restrict result, __m128i& totals) {
	totals = _mm_sub_epi16(_mm_add_epi16(totals, _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j) + 508)))), _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j) - 4))));
	totals = _mm_sub_epi16(_mm_add_epi16(totals, _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j + 1) + 508)))), _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j + 1) - 4))));
	__m128i shft = _mm_packus_epi16(_mm_srli_epi16(totals, 7), _mm_setzero_si128());
	if (single_last_column) {
		_mm_stream_si32(reinterpret_cast<int*>(result + 4 * (i*(width - 128) + j + 1)), _mm_extract_epi32(shft, 0));
	}
	else {
		_mm_stream_si64(reinterpret_cast<long long*>(result + 4 * (i*(width - 128) + j + 1)), _mm_extract_epi64(shft, 0));
	}
}

template<bool last_row>
void processRow(const uint8_t* const __restrict img, const int width, const int i, uint8_t* const __restrict result) {
	__m128i totals = _mm_setzero_si128();
	for (int j = 0; j < 128; ++j) {
		totals = _mm_add_epi16(totals, _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j)))));
	}
	__m128i shft = _mm_packus_epi16(_mm_srli_epi16(totals, 7), _mm_setzero_si128());
	_mm_stream_si64(reinterpret_cast<long long*>(result + 4 * (i*(width - 128))), _mm_extract_epi64(shft, 0));
	int j = 1;
	for (; j < width - 130; j += 2) {
		processCols<false>(img, width, i, j, result, totals);
	}
	if (j != width - 129) processCols<last_row>(img, width, i, j, result, totals);
}

void _boxBlur(const uint8_t* const __restrict img, const int width, const int start_row, const int rows, uint8_t* const __restrict result) {
	int i = start_row;
	for (; i < start_row + rows - 1; ++i) {
		processRow<false>(img, width, i, result);
	}
	processRow<true>(img, width, i, result);
}

void _boxBlurTransposable(const uint8_t* const __restrict img, const int width, const int start_row, const int rows, uint8_t* const __restrict result) {
	for (int i = start_row; i < start_row + rows; ++i) {
		__m128i totals = _mm_setzero_si128();
		for (int j = 0; j < 128; ++j) {
			totals = _mm_add_epi16(totals, _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j)))));
		}
		__m128i shft = _mm_packus_epi16(_mm_srli_epi16(totals, 7), _mm_setzero_si128());
		_mm_stream_si32(reinterpret_cast<int*>(result + 4 * (i*(width - 128))), _mm_extract_epi32(shft, 0));
		for (int j = 1; j < width - 128; ++j) {
			totals = _mm_sub_epi16(_mm_add_epi16(totals, _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j) + 508)))), _mm_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i* __restrict>(img + 4 * (i*width + j) - 4))));
			shft = _mm_packus_epi16(_mm_srli_epi16(totals, 7), _mm_setzero_si128());
			_mm_stream_si32(reinterpret_cast<int*>(result + 4 * (i*(width - 128) + j)), _mm_extract_epi32(shft, 0));
		}
	}
}

template<bool multithread>
void boxBlur(const uint8_t* const __restrict img, const int width, const int height, uint8_t* const __restrict result) {
	if (multithread) {
		const int32_t hw_concur = std::min(height >> 4, static_cast<int32_t>(std::thread::hardware_concurrency()));
		if (hw_concur > 1) {
			std::vector<std::future<void>> fut(hw_concur);
			const int thread_stride = (height - 1) / hw_concur + 1;
			int i = 0, start = 0;
			for (; i < std::min(height - 1, hw_concur - 1); ++i, start += thread_stride) {
				fut[i] = std::async(std::launch::async, _boxBlur, img, width, start, thread_stride, result);
			}
			fut[i] = std::async(std::launch::async, _boxBlur, img, width, start, height - start, result);
			for (int j = 0; j <= i; ++j) fut[j].wait();
		}
		else {
			_boxBlur(img, width, 0, height, result);
		}
	}
	else {
		_boxBlur(img, width, 0, height, result);
	}
}