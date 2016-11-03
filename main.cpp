/*******************************************************************
*   main.cpp
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

#include <chrono>
#include <cstdio>
#include <iostream>

#include "BoxBlur.h"

using namespace std::chrono;

int main() {
	constexpr int width = 1920;
	constexpr int height = 1080;
	constexpr int warmups = 300;
	constexpr int runs = 300;
	constexpr bool multithread = true;

	FILE* fp = fopen("test.bin", "rb");
	if (fp == nullptr) {
		std::cerr << "Failed to open test.bin. Aborting." << std::endl;
		return EXIT_FAILURE;
	}

	uint8_t* const img = new uint8_t[4*width*height];
	static_cast<void>(fread(img, 1, 4 * width*height, fp));

	uint8_t* const ref = new uint8_t[4 * (width - 128)*height];
	uint8_t* const result = new uint8_t[4 * (width-128)*height];

	_boxBlurref(img, width, 0, height, ref);

	for (int i = 0; i < warmups; ++i) boxBlur<multithread>(img, width, height, result);
	auto start = high_resolution_clock::now();
	for (int i = 0; i < runs; ++i) boxBlur<multithread>(img, width, height, result);
	auto end = high_resolution_clock::now();

	size_t checktotal = 0ULL;
	for (int i = 0; i < 4 * (width - 128)*height; ++i) {
		checktotal += ref[i];
	}

	size_t total = 0ULL;
	for (int i = 0; i < 4 * (width - 128)*height; ++i) {
		total += result[i];
	}
	std::cout << "Checksum: " << total << std::endl;

	for (int i = 0; i < 4 * (width - 128)*height; ++i) {
		if (ref[i] != result[i]) {
			std::cerr << "Disagreement at " << i << ". Expected " << +ref[i] << ", got " << +result[i] << std::endl;
			//return EXIT_FAILURE;
		}
	}

	if (total != checktotal) {
		std::cerr << "ERROR: BAD CHECKSUM!" << std::endl;
	}

	const double us = static_cast<double>(duration_cast<nanoseconds>(end - start).count()) / static_cast<double>(runs) * 1e-3;
	std::cout << "Time: " << us << " us." << std::endl;
}