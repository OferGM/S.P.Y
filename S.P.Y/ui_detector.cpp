//ui_detector
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
    // Preprocessing with improved parameters for both themes
    cv::Mat processed = image.clone();
    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);

    // Apply adaptive thresholding with optimal parameters
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    // Use different edge detection parameters based on theme
    cv::Mat edges;
    int lowThreshold = isDarkTheme ? 10 : 20;  // Lower thresholds to catch more edges
    int highThreshold = isDarkTheme ? 40 : 70;
    cv::Canny(gray, edges, lowThreshold, highThreshold);

    // Use larger structuring element for more robust edge connection
    cv::Mat dilatedEdges;
    cv::dilate(edges, dilatedEdges, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7))); // Increased from 5x5

    // Find contours with improved parameters
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilatedEdges, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<cv::Rect> inputFields;
    cv::Size imgSize = image.size();

    // More lenient field detection parameters
    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 50) continue;  // Even lower minimum area

        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;

        // More inclusive aspect ratio and size constraints
        if (rect.width > imgSize.width * 0.08 && // Even less restrictive width
            rect.height > 10 && rect.height < 120 && // Wider height range
            aspectRatio > 1.2 && aspectRatio < 30) { // More inclusive aspect ratio

            // Less restrictive position requirements
            bool isInFormPosition = rect.y > imgSize.height * 0.05 &&
                rect.y < imgSize.height * 0.95 &&
                rect.x > imgSize.width * 0.03 &&
                rect.x + rect.width < imgSize.width * 0.97;

            if (isInFormPosition) {
                inputFields.push_back(rect);
            }
        }
    }

    // Additional multi-scale detection for input fields
    // This helps catch fields that might be missed by edge detection
    std::vector<cv::Rect> additionalFields;

    // Method 1: Color-based detection
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);

    // Multiple threshold values to catch different field styles
    std::vector<int> threshValues;
    if (isDarkTheme) {
        threshValues = { 30, 50, 70 }; // Multiple thresholds for dark themes
    }
    else {
        threshValues = { 180, 200, 220 }; // Multiple thresholds for light themes
    }

    for (int threshValue : threshValues) {
        cv::Mat valueThresh;
        cv::threshold(channels[2], valueThresh, threshValue, 255,
            isDarkTheme ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV);

        // Clean up with morphological operations
        cv::Mat morphed;
        cv::morphologyEx(valueThresh, morphed, cv::MORPH_CLOSE,
            cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 5)));

        std::vector<std::vector<cv::Point>> colorContours;
        cv::findContours(morphed, colorContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : colorContours) {
            if (cv::contourArea(contour) < 50) continue;

            cv::Rect rect = cv::boundingRect(contour);
            double aspectRatio = static_cast<double>(rect.width) / rect.height;

            if (rect.width > imgSize.width * 0.08 &&
                rect.height > 10 && rect.height < 120 &&
                aspectRatio > 1.2 && aspectRatio < 30) {

                additionalFields.push_back(rect);
            }
        }
    }

    // Method 2: Saturation-based detection (often works well for input fields)
    cv::Mat satThresh;
    cv::threshold(channels[1], satThresh, 30, 255, cv::THRESH_BINARY_INV); // Low saturation areas

    cv::Mat satMorphed;
    cv::morphologyEx(satThresh, satMorphed, cv::MORPH_CLOSE,
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 5)));

    std::vector<std::vector<cv::Point>> satContours;
    cv::findContours(satMorphed, satContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : satContours) {
        if (cv::contourArea(contour) < 50) continue;

        cv::Rect rect = cv::boundingRect(contour);
        double aspectRatio = static_cast<double>(rect.width) / rect.height;

        if (rect.width > imgSize.width * 0.08 &&
            rect.height > 10 && rect.height < 120 &&
            aspectRatio > 1.2 && aspectRatio < 30) {

            additionalFields.push_back(rect);
        }
    }

    // Add additional fields that don't overlap significantly with existing ones
    for (const auto& newField : additionalFields) {
        bool isNovel = true;
        for (const auto& existing : inputFields) {
            // Calculate intersection over union
            cv::Rect intersection = existing & newField;
            double iou = intersection.area() / static_cast<double>((existing.area() + newField.area() - intersection.area()));

            if (iou > 0.2) { // Lower threshold to catch more potential overlaps
                isNovel = false;
                break;
            }
        }

        if (isNovel) {
            inputFields.push_back(newField);
        }
    }

    if (inputFields.size() < 2) {
        // Method 3: Look for rectangular areas with consistent brightness
        cv::Mat grayBlurred;
        cv::GaussianBlur(gray, grayBlurred, cv::Size(9, 9), 0);

        cv::Mat gradX, gradY;
        cv::Sobel(grayBlurred, gradX, CV_32F, 1, 0);
        cv::Sobel(grayBlurred, gradY, CV_32F, 0, 1);

        // Calculate magnitude of gradient
        cv::Mat gradMag;
        cv::magnitude(gradX, gradY, gradMag);

        // Normalize and convert to 8-bit
        cv::normalize(gradMag, gradMag, 0, 255, cv::NORM_MINMAX);
        cv::Mat gradMag8U;
        gradMag.convertTo(gradMag8U, CV_8U);

        // Threshold gradient magnitude to find edges
        cv::Mat gradThresh;
        cv::threshold(gradMag8U, gradThresh, 50, 255, cv::THRESH_BINARY);

        // Find contours of high gradient regions (potential field boundaries)
        std::vector<std::vector<cv::Point>> gradContours;
        cv::findContours(gradThresh, gradContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : gradContours) {
            if (cv::contourArea(contour) < 100) continue;

            // Approximate contour to get simpler polygon
            std::vector<cv::Point> approxCurve;
            cv::approxPolyDP(contour, approxCurve, 0.02 * cv::arcLength(contour, true), true);

            // Check if polygon has 4-6 sides (rectangular-ish)
            if (approxCurve.size() >= 4 && approxCurve.size() <= 6) {
                cv::Rect rect = cv::boundingRect(approxCurve);
                double aspectRatio = static_cast<double>(rect.width) / rect.height;

                if (rect.width > imgSize.width * 0.08 &&
                    rect.height > 10 && rect.height < 120 &&
                    aspectRatio > 1.2 && aspectRatio < 30) {

                    // Check if this is a novel field
                    bool isNovel = true;
                    for (const auto& existing : inputFields) {
                        double iou = (existing & rect).area() /
                            static_cast<double>((existing.area() + rect.area() - (existing & rect).area()));

                        if (iou > 0.2) {
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

    // Final step: Merge overlapping or very close rectangles
    if (!inputFields.empty()) {
        std::vector<cv::Rect> mergedFields;
        std::vector<bool> used(inputFields.size(), false);

        for (size_t i = 0; i < inputFields.size(); i++) {
            if (used[i]) continue;

            cv::Rect mergedRect = inputFields[i];
            used[i] = true;

            bool merged;
            do {
                merged = false;
                for (size_t j = 0; j < inputFields.size(); j++) {
                    if (used[j] || i == j) continue;

                    // Check if rects overlap or are very close
                    cv::Rect r1 = mergedRect;
                    cv::Rect r2 = inputFields[j];

                    // Expand rects slightly to catch nearby fields
                    r1.x -= 5; r1.y -= 5; r1.width += 10; r1.height += 10;
                    r2.x -= 5; r2.y -= 5; r2.width += 10; r2.height += 10;

                    if ((r1 & r2).area() > 0) {
                        // Merge the rectangles
                        mergedRect = mergedRect | inputFields[j];
                        used[j] = true;
                        merged = true;
                    }
                }
            } while (merged);

            mergedFields.push_back(mergedRect);
        }

        inputFields = mergedFields;
    }

    // Remove duplicates and sort by Y-coordinate (top to bottom)
    std::sort(inputFields.begin(), inputFields.end(),
        [](const cv::Rect& a, const cv::Rect& b) { return a.y < b.y; });

    return inputFields;
}