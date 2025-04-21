#pragma once
#include "common.h"
#include "ocr_processor.h"
#include "ui_detector.h"
#include "image_utils.h"

class LoginDetector {
public:
    LoginDetector();
    ~LoginDetector();

    // Sets the minimum confidence threshold for login detection.
    void setConfidenceThreshold(float threshold);

    // New enum for operation modes
    enum class OperationMode {
        DETECT_LOGIN,
        EXTRACT_FIELDS
    };

    // Structure to hold extracted field information
    struct ExtractedFields {
        std::string username;
        int passwordDots;
        bool usernameFieldPresent;
        bool passwordFieldPresent;
    };

    // detectLogin method with mode parameter
    bool detectLogin(const std::string& imagePath, OperationMode mode = OperationMode::DETECT_LOGIN);

    // method to extract username and password fields
    ExtractedFields extractLoginFields(const std::string& imagePath);

private:
    // Confidence threshold for login detection.
    float confidenceThreshold;

    // Sets of keywords used to determine login pages.
    std::unordered_set<std::string> loginKeywords;
    std::unordered_set<std::string> strongKeywords;

    OCRProcessor ocrProcessor;
    UIDetector uiDetector;

    float computeLoginConfidence(const std::string& recognizedText, const std::vector<WordBox>& words, bool isDarkTheme);

    ExtractedFields analyzeLoginFields(const cv::Mat& image,
        const std::vector<cv::Rect>& inputFields,
        const std::vector<WordBox>& words);

    int countPasswordDots(const cv::Mat& passwordField);
    std::string extractUsernameContent(const cv::Mat& usernameField, const std::vector<WordBox>& words);
};