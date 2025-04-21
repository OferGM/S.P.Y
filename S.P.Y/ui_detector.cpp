#include "ui_detector.h"

// Constructor for the UIDetector class
UIDetector::UIDetector() {}

//Detects login UI elements in an image

bool UIDetector::detectLoginUIElements(const cv::Mat& image, bool isDarkTheme) {
    // Preprocess the image to highlight UI elements
    cv::Mat processed = preprocessImage(image, isDarkTheme);
    cv::Size imgSize = processed.size();

    // Find contours in the processed image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(processed, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    // Store detected input fields
    std::vector<cv::Rect> inputFields;
    int totalInputFields = 0;
    int totalButtons = 0;

    // Initial pass to detect obvious input fields
    for (const auto& contour : contours) {
        // Skip small contours (noise)
        if (cv::contourArea(contour) < 100) continue;

        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;

        // Check if the rectangle has input field characteristics
        if (rect.width > imgSize.width * 0.15 && rect.height > 20 && rect.height < 80 &&
            aspectRatio > 2.5 && aspectRatio < 20) {

            // Check if the potential input field is in a typical form position
            bool isInFormPosition = rect.y > imgSize.height * 0.2 && rect.y < imgSize.height * 0.8 &&
                rect.x > imgSize.width * 0.1 && rect.x + rect.width < imgSize.width * 0.9;

            if (isInFormPosition) {
                totalInputFields++;
                inputFields.push_back(rect);
            }
        }
    }

    // If there are many contours, process them in parallel using multiple threads
    if (contours.size() > 500) {
        const size_t maxThreads = NUM_THREADS; // Defined elsewhere
        const size_t contoursSize = contours.size();

        // Calculate how many threads to use (min of maxThreads or number of contours/100)
        const size_t numThreads = (contoursSize / 100 + 1) < maxThreads ?
            (contoursSize / 100 + 1) : maxThreads;

        const size_t contoursPerThread = contoursSize / numThreads;
        std::vector<std::future<std::pair<int, int>>> futures;
        futures.reserve(numThreads);

        // Distribute the contour processing across multiple threads
        for (size_t idx = 0; idx < numThreads; idx++) {
            size_t startIdx = idx * contoursPerThread;
            size_t endIdx = (idx == numThreads - 1) ? contoursSize : (idx + 1) * contoursPerThread;
            futures.push_back(std::async(std::launch::async,
                [this, &contours, &imgSize, &inputFields, startIdx, endIdx]() {
                    return processContours(contours, imgSize, inputFields, startIdx, endIdx);
                }));
        }

        // Collect results from all threads
        for (auto& fut : futures) {
            auto result = fut.get();
            totalInputFields += result.first;
            totalButtons += result.second;
        }
    }
    else {
        // For smaller number of contours, process sequentially
        auto result = processContours(contours, imgSize, inputFields, 0, contours.size());
        totalInputFields += result.first;
        totalButtons += result.second;
    }

    // Log the detection results
    Logger::log(Logger::Level::INFO, "UI Detection: " + std::to_string(totalInputFields) + " input fields, " +
        std::to_string(totalButtons) + " buttons");

    // Return true if login form is detected (different criteria based on elements found)
    return (totalInputFields >= 1 && totalButtons >= 1) || (totalInputFields >= 2);
}

//Process a subset of contours to identify UI elements
std::pair<int, int> UIDetector::processContours(const std::vector<std::vector<cv::Point>>& contours,
    const cv::Size& imgSize,
    const std::vector<cv::Rect>& knownInputFields,
    size_t startIdx,
    size_t endIdx) {
    int inputFieldCount = 0;
    int buttonCount = 0;

    // Process each contour in the assigned range
    for (size_t idx = startIdx; idx < endIdx; idx++) {
        const auto& contour = contours[idx];
        if (cv::contourArea(contour) < 100) continue; // Skip small contours

        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;

        // Check for input field characteristics
        if (rect.width > imgSize.width * 0.15 && rect.height > 20 && rect.height < 80 &&
            aspectRatio > 2.5 && aspectRatio < 20) {
            bool isInFormPosition = rect.y > imgSize.height * 0.2 && rect.y < imgSize.height * 0.8 &&
                rect.x > imgSize.width * 0.1 && rect.x + rect.width < imgSize.width * 0.9;
            if (isInFormPosition) inputFieldCount++;
        }

        // Check for button characteristics
        if (rect.width > imgSize.width * 0.1 && rect.height > 20 && rect.height < 70 &&
            aspectRatio > 1.5 && aspectRatio < 8) {
            bool isButtonPosition = false;

            // Check if this potential button is positioned below any known input field
            for (const auto& field : knownInputFields) {
                if ((rect.y > field.y + field.height) &&
                    std::abs((rect.x + rect.width / 2) - (field.x + field.width / 2)) < field.width) {
                    isButtonPosition = true;
                    break;
                }
            }
            if (isButtonPosition) buttonCount++;
        }
    }
    return std::make_pair(inputFieldCount, buttonCount);
}

//Preprocess the image to highlight UI elements
cv::Mat UIDetector::preprocessImage(const cv::Mat& image, bool isDarkTheme) {
    cv::Mat processed = image.clone();

    // Convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    // For dark themes, normalize the image to improve contrast
    if (isDarkTheme) cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX);

    // Apply Gaussian blur to reduce noise
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // Detect edges using Canny algorithm (with different thresholds for dark/light themes)
    cv::Mat edges;
    cv::Canny(gray, edges, isDarkTheme ? 20 : 30, isDarkTheme ? 60 : 90);

    // Dilate the edges to connect nearby edges
    cv::Mat dilatedEdges;
    cv::dilate(edges, dilatedEdges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));

    return dilatedEdges;
}

std::vector<cv::Rect> UIDetector::detectInputFields(const cv::Mat& image, bool isDarkTheme) {
    // Preprocessing with optimized parameters
    cv::Mat processed = image.clone();
    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    // Fast blur for noise reduction - replaced Gaussian with median for speed
    cv::medianBlur(gray, gray, 5);

    // Optimized edge detection with safe parameters
    cv::Mat edges;
    int lowThreshold = isDarkTheme ? 20 : 30;
    int highThreshold = isDarkTheme ? 60 : 90;
    cv::Canny(gray, edges, lowThreshold, highThreshold);

    // More efficient dilation with smaller kernel
    cv::Mat dilatedEdges;
    cv::dilate(edges, dilatedEdges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));

    // Find contours - using RETR_EXTERNAL is faster as it only gets outer contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilatedEdges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> inputFields;
    cv::Size imgSize = image.size();

    // Fast filtering of contours
    for (const auto& contour : contours) {
        // Quick area check first to skip small contours faster
        double area = cv::contourArea(contour);
        if (area < 100 || area > imgSize.area() * 0.2) continue;

        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;

        // Field heuristics
        if (rect.width > imgSize.width * 0.1 &&
            rect.height > 15 && rect.height < 100 &&
            aspectRatio > 1.5 && aspectRatio < 20) {

            // Simple position check
            if (rect.y > imgSize.height * 0.1 &&
                rect.y < imgSize.height * 0.9) {
                inputFields.push_back(rect);
            }
        }
    }

    // If we found enough fields, don't run the slower methods
    if (inputFields.size() >= 2) {
        return inputFields;
    }

    // Second method: direct binary thresholding - very fast
    cv::Mat binaryImage;

    // Adapt threshold based on theme
    int threshVal = isDarkTheme ? 60 : 200;
    cv::threshold(gray, binaryImage, threshVal, 255, isDarkTheme ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV);

    // Find contours in binary image
    std::vector<std::vector<cv::Point>> binaryContours;
    cv::findContours(binaryImage, binaryContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : binaryContours) {
        double area = cv::contourArea(contour);
        if (area < 100 || area > imgSize.area() * 0.2) continue;

        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;

        if (rect.width > imgSize.width * 0.1 &&
            rect.height > 15 && rect.height < 100 &&
            aspectRatio > 1.5 && aspectRatio < 20) {

            // Check if this is a novel field
            bool isNovel = true;
            for (const auto& existingField : inputFields) {
                double iou = (existingField & rect).area() / static_cast<double>((existingField.area() + rect.area() - (existingField & rect).area()));
                if (iou > 0.3) {
                    isNovel = false;
                    break;
                }
            }

            if (isNovel) {
                inputFields.push_back(rect);
            }
        }
    }

    // If still not enough fields, look for rectangular shapes
    if (inputFields.size() < 2) {
        // Use rectangle detection instead of HoughCircles
        std::vector<std::vector<cv::Point>> approxContours;

        for (const auto& contour : contours) {
            if (cv::contourArea(contour) < 100) continue;

            // Approximate contour to detect rectangular shapes
            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 0.04 * cv::arcLength(contour, true), true);

            // Check if it has 4-6 sides (rectangular)
            if (approx.size() >= 4 && approx.size() <= 6) {
                cv::Rect rect = cv::boundingRect(approx);
                double aspectRatio = static_cast<double>(rect.width) / rect.height;

                if (rect.width > imgSize.width * 0.1 &&
                    rect.height > 15 && rect.height < 100 &&
                    aspectRatio > 1.5 && aspectRatio < 20) {

                    bool isNovel = true;
                    for (const auto& existingField : inputFields) {
                        double iou = (existingField & rect).area() / static_cast<double>((existingField.area() + rect.area() - (existingField & rect).area()));
                        if (iou > 0.3) {
                            isNovel = false;
                            break;
                        }
                    }

                    if (isNovel) {
                        inputFields.push_back(rect);
                    }
                }
            }
        }
    }

    // Sort by Y-coordinate
    std::sort(inputFields.begin(), inputFields.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.y < b.y; });

    return inputFields;
}