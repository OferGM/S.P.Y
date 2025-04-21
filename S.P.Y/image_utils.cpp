#include "image_utils.h"

//Detects whether an image has a dark theme or light theme
bool ImageUtils::detectTheme(const cv::Mat& image) {
    // Convert image to grayscale for brightness analysis
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Method 1: Check average brightness of the entire image
    // Values below 128 (on 0-255 scale) suggest a dark theme
    cv::Scalar meanIntensity = cv::mean(gray);
    bool darkByBrightness = meanIntensity[0] < 128;

    // Method 2: Calculate ratio of dark to light pixels
    // First threshold the image to separate dark from light pixels
    cv::Mat thresholded;
    cv::threshold(gray, thresholded, 128, 255, cv::THRESH_BINARY);

    // Count dark pixels and calculate their proportion of the total image
    int darkPixels = gray.rows * gray.cols - cv::countNonZero(thresholded);
    float darkRatio = static_cast<float>(darkPixels) / (gray.rows * gray.cols);
    bool darkByPixelRatio = darkRatio > 0.6; // If >60% of pixels are dark, likely dark theme

    // Method 3: Analyze header and footer regions
    // Many applications have dark headers/footers even in light themes
    cv::Rect topArea(0, 0, gray.cols, gray.rows * 0.1); // Top 10% of image
    cv::Rect bottomArea(0, gray.rows * 0.9, gray.cols, gray.rows * 0.1); // Bottom 10% of image
    cv::Mat topROI = gray(topArea);
    cv::Mat bottomROI = gray(bottomArea);

    // Lower threshold (100) is used for header/footer regions to better detect
    // dark UI elements against potentially light backgrounds
    bool darkHeader = cv::mean(topROI)[0] < 100;
    bool darkFooter = cv::mean(bottomROI)[0] < 100;

    // Combine all methods using a weighted scoring system
    // More weight is given to overall brightness and pixel ratio (2 points each)
    // Less weight to header/footer areas (1 point each)
    int darkScore = (darkByBrightness ? 2 : 0) + (darkByPixelRatio ? 2 : 0) +
        (darkHeader ? 1 : 0) + (darkFooter ? 1 : 0);
    bool isDarkTheme = darkScore >= 3; // Threshold for dark theme determination

    // Log the result for debugging and metrics tracking
    Logger::log(Logger::Level::INFO, "Image appears to be " + std::string(isDarkTheme ? "dark" : "light") + " themed");

    return isDarkTheme;
}

//Validates if a file exists and can be opened as an image

bool ImageUtils::isValidImageFile(const std::string& filePath) {
    // First check: verify file exists and is accessible
    std::ifstream f(filePath, std::ios::binary);
    if (!f.good()) {
        Logger::log(Logger::Level::ERROR, "File does not exist: " + filePath);
        return false;
    }
    f.close();

    // Second check: verify OpenCV can load it as an image
    cv::Mat img = cv::imread(filePath);
    if (img.empty()) {
        Logger::log(Logger::Level::ERROR, "Could not load image: " + filePath);
        return false;
    }

    return true;
}