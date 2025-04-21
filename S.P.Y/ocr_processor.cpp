#include "ocr_processor.h"

OCRProcessor::OCRProcessor() {
    // Initialize dictionary of login-related keywords and phrases
    // These words help identify if text on a screen is related to authentication processes
    loginKeywords = {
        // Basic login terms
        "login", "sign in", "signin", "log in", "username", "password", "email",
        "phone", "forgot password", "reset password", "remember me", "create account",
        // Account creation and registration terms
        "register", "authentication", "verify", "credentials", "account",
        "welcome back", "sign up", "signup", "continue with", "continue", "email address",
        "don't have an account", "new account", "create your account", "join now",
        // Social login options
        "continue with google", "continue with microsoft", "continue with apple",
        "continue with facebook", "sign in with google", "sign in with apple",
        "facebook", "google", "apple", "microsoft", "steam", "epic games",
        // Legal and policy references often found on login screens
        "privacy policy", "terms of service", "terms of use", "terms and conditions",
        // Action buttons typically found on login forms
        "next", "submit", "go", "enter", "send code", "verify email", "get started",
        // Form and field related terms
        "required", "required field", "remember this device", "keep me signed in",
        "stay signed in", "keep me logged in", "not your computer", "guest mode"
    };
}

//Destructor for OCRProcessor
OCRProcessor::~OCRProcessor() {}

// Creates or retrieves a thread-local Tesseract OCR engine instance
tesseract::TessBaseAPI* OCRProcessor::getThreadLocalOCR() {
    // 'thread_local' ensures that each thread has its own separate instance
    // This avoids thread contention when multiple threads perform OCR simultaneously
    thread_local tesseract::TessBaseAPI* localOCR = nullptr;

    // Initialize the OCR engine if it hasn't been created for this thread yet
    if (!localOCR) {
        localOCR = new tesseract::TessBaseAPI();

        // Initialize Tesseract with English language and LSTM OCR engine mode
        // NULL for datapath means use default location for tessdata
        if (localOCR->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY)) {
            Logger::log(Logger::Level::ERROR, "Failed to initialize Tesseract OCR engine");
            exit(1);
        }

        // Configure Tesseract parameters for optimal text recognition in UI contexts

        // Set page segmentation mode to automatic detection
        localOCR->SetPageSegMode(tesseract::PSM_AUTO);

        // Disable automatic image inversion - we handle this manually with image variants
        localOCR->SetVariable("tessedit_do_invert", "0");

        // Performance optimizations to speed up processing
        localOCR->SetVariable("textord_fast_pitch_test", "1");
        localOCR->SetVariable("textord_max_fixtures", "1");

        // Exclude special characters that are rarely relevant in UI text and may cause false positives
        localOCR->SetVariable("tessedit_char_blacklist", "{}[]()^*;~`|\\");

        // Disable creation of HOCR and box files to improve performance
        localOCR->SetVariable("tessedit_create_hocr", "0");
        localOCR->SetVariable("tessedit_create_boxfile", "0");

        // Use Otsu's method for image thresholding (adaptive binarization)
        localOCR->SetVariable("thresholding_method", "2");

        // Redirect debug output to /dev/null to avoid log file creation
        localOCR->SetVariable("debug_file", "/dev/null");

        // Additional fine-tuning parameters for character recognition
        localOCR->SetVariable("classify_bln_numeric_mode", "0");
        localOCR->SetVariable("edges_max_children_per_outline", "40");
        localOCR->SetVariable("edges_children_count_limit", "5");
    }
    return localOCR;
}

// Creates multiple processed versions of an image to improve OCR accuracy
std::vector<cv::Mat> OCRProcessor::generateImageVariants(const cv::Mat& originalImage, bool isDarkTheme) {
    std::vector<cv::Mat> processedImages;

    // Make a copy of the original image and resize if necessary
    cv::Mat baseImage = originalImage.clone();
    if (baseImage.cols > 1800 || baseImage.rows > 1800) {
        // Resize large images to improve processing speed while maintaining quality
        float scale = 1800.0f / std::max(baseImage.cols, baseImage.rows);
        cv::resize(baseImage, baseImage, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // Convert image to grayscale (required for most OCR preprocessing techniques)
    cv::Mat grayImage;
    cv::cvtColor(baseImage, grayImage, cv::COLOR_BGR2GRAY);

    // Apply Gaussian blur to reduce noise and improve edge detection
    cv::Mat standardProcessed;
    cv::GaussianBlur(grayImage, standardProcessed, cv::Size(3, 3), 0);

    // Apply adaptive thresholding to handle varying illumination conditions
    cv::Mat adaptiveThresh;
    cv::adaptiveThreshold(standardProcessed, adaptiveThresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);

    // Add standard processed image to results
    processedImages.push_back(standardProcessed);

    // Create additional variants based on the UI theme
    if (isDarkTheme) {
        // For dark themes, create inverted variants to better extract light text on dark backgrounds
        cv::Mat inverted;
        cv::bitwise_not(standardProcessed, inverted);
        processedImages.push_back(inverted);

        // Apply histogram equalization to the inverted image to enhance contrast
        cv::Mat enhancedInverted;
        cv::equalizeHist(inverted, enhancedInverted);
        processedImages.push_back(enhancedInverted);
    }
    else {
        // For light themes, apply enhancement techniques to improve text visibility
        cv::Mat enhanced;
        cv::equalizeHist(standardProcessed, enhanced);
        processedImages.push_back(enhanced);

        // Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) 
        // for better local contrast enhancement
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
        cv::Mat claheImage;
        clahe->apply(standardProcessed, claheImage);
        processedImages.push_back(claheImage);
    }

    // Add the adaptive threshold image to the results
    processedImages.push_back(adaptiveThresh);

    return processedImages;
}

//Performs OCR on multiple image variants and selects the best result
std::pair<std::string, std::vector<WordBox>> OCRProcessor::performEnhancedOCR(const cv::Mat& originalImage, bool isDarkTheme) {
    // Generate multiple processed versions of the image
    std::vector<cv::Mat> processedImages = generateImageVariants(originalImage, isDarkTheme);

    // Launch parallel OCR tasks for each processed image variant
    std::vector<std::future<std::pair<std::string, std::vector<WordBox>>>> futures;
    futures.reserve(processedImages.size());

    // Process each image variant in parallel using async tasks
    for (const auto& img : processedImages) {
        futures.push_back(std::async(std::launch::async, [this, img]() {
            // Convert OpenCV Mat to Leptonica PIX format required by Tesseract
            PIX* pix = pixCreate(img.cols, img.rows, 8);
            if (!pix) {
                Logger::log(Logger::Level::ERROR, "Failed to create PIX structure");
                return std::make_pair(std::string(""), std::vector<WordBox>());
            }

            // Copy pixel data from OpenCV Mat to Leptonica PIX
            for (int y_cords = 0; y_cords < img.rows; y_cords++) {
                for (int x_cords = 0; x_cords < img.cols; x_cords++) {
                    pixSetPixel(pix, x_cords, y_cords, img.at<uchar>(y_cords, x_cords));
                }
            }

            // Get thread-local OCR instance and set the image for processing
            tesseract::TessBaseAPI* localOCR = getThreadLocalOCR();
            localOCR->SetImage(pix);

            // Perform OCR recognition
            localOCR->Recognize(0);

            // Extract the recognized text
            char* text = localOCR->GetUTF8Text();
            std::string recognizedText = text ? text : "";
            delete[] text;  // Free memory allocated by Tesseract

            // Convert text to lowercase for case-insensitive keyword matching
            std::transform(recognizedText.begin(), recognizedText.end(), recognizedText.begin(),
                [](unsigned char c) { return std::tolower(c); });

            // Extract individual words with their bounding boxes
            std::vector<WordBox> words = getWordsAndBoxes(localOCR);

            // Clean up Leptonica PIX structure
            pixDestroy(&pix);

            return std::make_pair(recognizedText, words);
            }));
    }

    // Collect OCR results from all parallel tasks
    std::vector<std::string> textResults;
    std::vector<std::vector<WordBox>> wordsResults;

    textResults.reserve(futures.size());
    wordsResults.reserve(futures.size());

    for (auto& fut : futures) {
        auto result = fut.get();
        textResults.push_back(result.first);
        wordsResults.push_back(result.second);
    }

    // Select the best OCR result based on the number of login keywords found
    int bestIndex = 0;
    int maxKeywords = -1;

    for (size_t idx = 0; idx < textResults.size(); idx++) {
        int keywordCount = countKeywords(textResults[idx]);
        if (keywordCount > maxKeywords) {
            maxKeywords = keywordCount;
            bestIndex = idx;
        }
    }

    Logger::log(Logger::Level::INFO, "Best OCR method found " + std::to_string(maxKeywords) + " keywords");

    // If no login keywords found in any variant, combine all results
    if (maxKeywords == 0) {
        std::string combinedText;
        std::vector<WordBox> combinedWords;

        for (size_t idx = 0; idx < textResults.size(); idx++) {
            combinedText += textResults[idx] + " ";
            combinedWords.insert(combinedWords.end(), wordsResults[idx].begin(), wordsResults[idx].end());
        }
        return { combinedText, combinedWords };
    }

    // Return the best result based on keyword count
    return { textResults[bestIndex], wordsResults[bestIndex] };
}

//Extracts individual words and their bounding boxes from OCR results
std::vector<WordBox> OCRProcessor::getWordsAndBoxes(tesseract::TessBaseAPI* localOCR) {
    std::vector<WordBox> word_boxes;

    // Get an iterator over the OCR results to access individual words
    tesseract::ResultIterator* ri = localOCR->GetIterator();

    if (ri) {
        do {
            // Skip if the current element is not a word
            if (ri->Empty(tesseract::RIL_WORD)) continue;

            // Get the word text
            const char* word = ri->GetUTF8Text(tesseract::RIL_WORD);

            if (word && strlen(word) > 0) {
                // Get the word's bounding box coordinates
                int x1, y1, x2, y2;
                ri->BoundingBox(tesseract::RIL_WORD, &x1, &y1, &x2, &y2);

                // Get the recognition confidence (0-100)
                float conf = ri->Confidence(tesseract::RIL_WORD);

                // Convert word to lowercase for consistent processing
                std::string words(word);
                std::transform(words.begin(), words.end(), words.begin(), ::tolower);

                // Filter words by confidence threshold and valid dimensions
                // This helps eliminate low-confidence OCR results that might be errors
                if (conf > 30 && words.length() > 1 && x2 > x1 && y2 > y1) {
                    word_boxes.emplace_back(words, cv::Rect(x1, y1, x2 - x1, y2 - y1), conf);
                }

                delete[] word;  // Free memory allocated by Tesseract
            }
        } while (ri->Next(tesseract::RIL_WORD));  // Move to the next word

        delete ri;  // Free the result iterator
    }
    return word_boxes;
}

//Counts the number of login-related keywords found in text

int OCRProcessor::countKeywords(const std::string& text) {
    int count = 0;

    // Search for each keyword in the recognized text
    for (const auto& keyword : loginKeywords) {
        size_t pos = 0;

        // Find all occurrences of the current keyword
        while ((pos = text.find(keyword, pos)) != std::string::npos) {
            count++;
            Logger::log(Logger::Level::DEBUG, "Found keyword: " + keyword);
            pos += keyword.length();  // Move past the current match
        }
    }

    return count;
}

// Main interface method for processing an image with OCR
std::pair<std::string, std::vector<WordBox>> OCRProcessor::processImage(const cv::Mat& image, bool isDarkTheme) {
    return performEnhancedOCR(image, isDarkTheme);
}