#pragma once
#include "common.h"

class OCRProcessor {
public:
    OCRProcessor();
    ~OCRProcessor();

    //Processes an image using OCR and returns recognized text and word boxes.
    std::pair<std::string, std::vector<WordBox>> processImage(const cv::Mat& image, bool isDarkTheme);

private:
    tesseract::TessBaseAPI* getThreadLocalOCR();
    std::vector<WordBox> getWordsAndBoxes(tesseract::TessBaseAPI* api);
    std::pair<std::string, std::vector<WordBox>> performEnhancedOCR(const cv::Mat& originalImage, bool isDarkTheme);
    std::vector<cv::Mat> generateImageVariants(const cv::Mat& originalImage, bool isDarkTheme);
    int countKeywords(const std::string& text);

    std::unordered_set<std::string> loginKeywords;
};