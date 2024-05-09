/**
 * @file main.cpp
 * @author Martin Kubicka (xkubic45@stud.fit.vutbr.cz)
 * @date 2024-04-14
 * 
 *  @brief Code for converting equirectangular projection to stereographic. 
 *         Credit goes to author of: https://github.com/chinhsuanwu/360-converter, which is used in this code and inspired by.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#define CONVERTER_IMPLEMENTATION
#include "./converter.hpp"

#ifndef CHANNEL_NUM
#define CHANNEL_NUM 1
#endif

#include <iostream>
#include <cstring> 

int main(int argc, char **argv)
{
	Converter::Image img;
	
	// input dir string
	const char* inputDir = "";
	size_t totalLength = strlen(inputDir) + strlen(argv[1]) + 1;
    char* inputPath = new char[totalLength];
    strcpy(inputPath, inputDir);
    strcat(inputPath, argv[1]);

	// output dir string
	const char* outputDir = argv[3]; 
	std::string fullPath(argv[1]);
	size_t lastSlashPos = fullPath.find_last_of("/\\");
    std::string fileName = fullPath.substr(lastSlashPos + 1);
	std::string combinedPath = std::string(outputDir) + fileName;
    const char* outputPath = combinedPath.c_str();

	// loading equirectangular input
	int w, h, bpp;
	img.img = stbi_load(inputPath, &w, &h, &bpp, CHANNEL_NUM);
	img.w = w, img.h = h;

	Converter::Equi equi = Converter::Equi(img);

	// converting to stereographic projection
	img = equi.toStereo().getStereo();
	stbi_write_png(outputPath, img.w, img.h, CHANNEL_NUM, img.img, img.w * CHANNEL_NUM);

	delete[] inputPath;

	return 0;
}

/*** End of main.py ***/
