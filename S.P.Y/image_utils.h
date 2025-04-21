//image_utils.h
#pragma once
#include "common.h"

class ImageUtils {
public:
    static bool detectTheme(const cv::Mat& image);
    static bool isValidImageFile(const std::string& filePath);
};