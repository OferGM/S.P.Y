#pragma once
#include "common.h"

class UIDetector {
public:
    UIDetector();
    // Detects UI elements indicative of a login screen.
    bool detectLoginUIElements(const cv::Mat& image, bool isDarkTheme);

    std::vector<cv::Rect> detectInputFields(const cv::Mat& image, bool isDarkTheme);
private:
    cv::Mat preprocessImage(const cv::Mat& image, bool isDarkTheme);
    std::pair<int, int> processContours(const std::vector<std::vector<cv::Point>>& contours,
        const cv::Size& imgSize,
        const std::vector<cv::Rect>& knownInputFields,
        size_t startIdx,
        size_t endIdx);
};