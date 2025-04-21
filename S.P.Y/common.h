#pragma once

// Standard library includes
#include <algorithm>
#include <chrono>
#include <fstream>
#include <future>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// OpenCV includes
#include <opencv2/opencv.hpp>

// Tesseract includes
#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

// Directory handling for Windows
#ifdef _WIN32
#include <direct.h>
#define GetCurrentDir _getcwd
#else
#include <unistd.h>
#define GetCurrentDir getcwd
#endif

// Number of threads based on hardware concurrency
const unsigned int NUM_THREADS = std::thread::hardware_concurrency() > 0 ? std::thread::hardware_concurrency() : 4;

//Utility class for logging messages at different levels.
class Logger {
public:
    enum class Level { DEBUG, INFO, WARNING, ERROR };

    static void log(Level level, const std::string& message) {
        static const char* levelStrings[] = { "DEBUG", "INFO", "WARNING", "ERROR" };
        std::cout << "[" << levelStrings[static_cast<int>(level)] << "] " << message << std::endl;
    }
};

struct WordBox {
    std::string word;   // Recognized text
    cv::Rect box;       // Bounding box around the word
    float confidence;   // Confidence percentage (0-100)

    WordBox(const std::string& w, const cv::Rect& b, float conf) : word(w), box(b), confidence(conf) {}
    WordBox() : word(""), confidence(0.0f) {}
};