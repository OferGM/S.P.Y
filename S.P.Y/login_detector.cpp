//login_detector.cpp
#include "login_detector.h"

LoginDetector::LoginDetector() : confidenceThreshold(0.35f) {
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

    // Initialize a list of high-value keywords that strongly indicate a login page
    // These keywords have higher weight in confidence calculation
    strongKeywords = {
        "sign in with", "sign in to", "log in to", "email address", "password",
        "username and password", "forgot password", "create account", "sign up",
        "continue with google", "continue with microsoft", "continue with apple",
        "remember me", "email or phone", "username", "login", "signin", "sign in",
        "log in", "create your account", "verify your identity", "required field"
    };
}

LoginDetector::~LoginDetector() {}

// Sets the confidence threshold for login screen detection

void LoginDetector::setConfidenceThreshold(float threshold) {
    confidenceThreshold = threshold;
}

// Main method to detect if an image contains a login screen
bool LoginDetector::detectLogin(const std::string& imagePath, OperationMode mode) {
    // If mode is EXTRACT_FIELDS, we still want to perform detection first
    // to leverage existing detection logic

    // Verify that the image file exists and is valid
    if (!ImageUtils::isValidImageFile(imagePath)) return false;

    // Load the image and detect if it has a dark theme
    cv::Mat originalImage = cv::imread(imagePath);
    bool isDarkTheme = ImageUtils::detectTheme(originalImage);

    // Run OCR and UI detection in parallel for better performance
    auto ocrFuture = std::async(std::launch::async, [&]() {
        return ocrProcessor.processImage(originalImage, isDarkTheme);
        });
    auto uiFuture = std::async(std::launch::async, [&]() {
        return uiDetector.detectLoginUIElements(originalImage, isDarkTheme);
        });

    // Wait for both processes to complete and get results
    auto ocrResult = ocrFuture.get();  // Contains recognized text and word boxes
    bool hasLoginUI = uiFuture.get();  // True if UI elements suggest a login form

    // Calculate confidence score based on OCR results and theme
    float confidence = computeLoginConfidence(ocrResult.first, ocrResult.second, isDarkTheme);

    // Final determination: high confidence AND UI elements consistent with login form
    bool isLoginScreen = (confidence > confidenceThreshold) && hasLoginUI;

    // Log the detection results
    Logger::log(Logger::Level::INFO, "Text confidence: " + std::to_string(confidence) +
        ", UI detection: " + std::string(hasLoginUI ? "true" : "false"));

    return isLoginScreen;
}

LoginDetector::ExtractedFields LoginDetector::extractLoginFields(const std::string& imagePath) {
    // Default return structure with empty values
    ExtractedFields fields;
    fields.username = "";
    fields.passwordDots = 0;
    fields.usernameFieldPresent = false;
    fields.passwordFieldPresent = false;

    // Verify that the image file exists and is valid
    if (!ImageUtils::isValidImageFile(imagePath)) return fields;

    // Load the image and detect if it has a dark theme
    cv::Mat originalImage = cv::imread(imagePath);
    bool isDarkTheme = ImageUtils::detectTheme(originalImage);

    // Log theme detection for debugging
    Logger::log(Logger::Level::INFO, std::string("Image appears to be ") +
        (isDarkTheme ? "dark" : "light") + " themed");

    // First, check if this is actually a login page
    bool isLoginPage = detectLogin(imagePath, OperationMode::DETECT_LOGIN);

    if (!isLoginPage) {
        Logger::log(Logger::Level::INFO, "Image doesn't appear to be a login page, but proceeding with field extraction anyway");
        // Continue anyway - we'll still try to extract fields even if confidence is low
    }

    // Process the image with OCR
    auto ocrResult = ocrProcessor.processImage(originalImage, isDarkTheme);
    Logger::log(Logger::Level::INFO, "OCR processing completed");

    // Detect input fields in the image with enhanced detection
    std::vector<cv::Rect> inputFields = uiDetector.detectInputFields(originalImage, isDarkTheme);
    Logger::log(Logger::Level::INFO, "Detected " + std::to_string(inputFields.size()) + " input fields");

    // If no input fields found, try with adjusted parameters
    if (inputFields.empty()) {
        Logger::log(Logger::Level::INFO, "No input fields detected, trying with adjusted parameters");

        // Create a copy with adjusted contrast to improve field detection
        cv::Mat enhancedImage;
        double contrastAlpha = isDarkTheme ? 1.3 : 1.2; // Increase contrast
        int brightnessBeta = isDarkTheme ? 10 : -10;    // Adjust brightness

        originalImage.convertTo(enhancedImage, -1, contrastAlpha, brightnessBeta);

        // Try detecting fields on enhanced image
        inputFields = uiDetector.detectInputFields(enhancedImage, isDarkTheme);
        Logger::log(Logger::Level::INFO, "Detected " + std::to_string(inputFields.size()) +
            " input fields after enhancement");
    }

    // If still no input fields found, return empty result
    if (inputFields.empty()) {
        Logger::log(Logger::Level::INFO, "No input fields detected for extraction");
        return fields;
    }

    // Analyze the detected fields and extract username/password info
    fields = analyzeLoginFields(originalImage, inputFields, ocrResult.second);

    // Log extraction results
    Logger::log(Logger::Level::INFO, "Username field present: " +
        std::string(fields.usernameFieldPresent ? "true" : "false"));
    Logger::log(Logger::Level::INFO, "Username content: " + fields.username);
    Logger::log(Logger::Level::INFO, "Password field present: " +
        std::string(fields.passwordFieldPresent ? "true" : "false"));
    Logger::log(Logger::Level::INFO, "Password dots count: " + std::to_string(fields.passwordDots));

    return fields;
}

LoginDetector::ExtractedFields LoginDetector::analyzeLoginFields(
    const cv::Mat& image,
    const std::vector<cv::Rect>& inputFields,
    const std::vector<WordBox>& words) {

    ExtractedFields fields;
    fields.username = "";
    fields.passwordDots = 0;
    fields.usernameFieldPresent = false;
    fields.passwordFieldPresent = false;

    if (inputFields.empty()) {
        return fields;
    }

    // More sophisticated field classification with improved parameters
    std::vector<std::pair<int, double>> usernameScores;  // <field index, score>
    std::vector<std::pair<int, double>> passwordScores;  // <field index, score>

    // Expanded search radius for field labels
    const int VERTICAL_SEARCH_RADIUS = 80;  // Increased from 50
    const int HORIZONTAL_SEARCH_RADIUS = 200;  // Increased from 150

    // For each field, calculate scores for being username vs password
    for (size_t i = 0; i < inputFields.size(); i++) {
        cv::Rect field = inputFields[i];
        double usernameScore = 0.0;
        double passwordScore = 0.0;

        // Check for prefilled content - common in username fields
        cv::Mat fieldRegion = image(field);
        cv::Mat fieldGray;
        cv::cvtColor(fieldRegion, fieldGray, cv::COLOR_BGR2GRAY);
        cv::Scalar meanIntensity = cv::mean(fieldGray);
        bool hasContent = meanIntensity[0] < 240 && meanIntensity[0] > 30; // Not too bright or dark
        if (hasContent) {
            usernameScore += 1.5; // Boost score for fields with content
        }

        // Check nearby labels with expanded search area
        for (const auto& word : words) {
            // Check if word is near the field (expanded proximity)
            bool isAboveField = (word.box.y + word.box.height <= field.y + VERTICAL_SEARCH_RADIUS) &&
                std::abs((word.box.x + word.box.width / 2) - (field.x + field.width / 2)) < HORIZONTAL_SEARCH_RADIUS;

            bool isLeftOfField = (word.box.x + word.box.width <= field.x + HORIZONTAL_SEARCH_RADIUS) &&
                (std::abs((word.box.y + word.box.height / 2) - (field.y + field.height / 2)) < VERTICAL_SEARCH_RADIUS);

            bool isBelowField = (word.box.y >= field.y + field.height - 5) &&
                (word.box.y <= field.y + field.height + VERTICAL_SEARCH_RADIUS) &&
                std::abs((word.box.x + word.box.width / 2) - (field.x + field.width / 2)) < HORIZONTAL_SEARCH_RADIUS;

            if (isAboveField || isLeftOfField || isBelowField) {
                std::string lowercaseWord = word.word;
                std::transform(lowercaseWord.begin(), lowercaseWord.end(), lowercaseWord.begin(), ::tolower);

                // Username indicators with adjusted weights
                if (lowercaseWord.find("user") != std::string::npos) usernameScore += 4.0;  
                if (lowercaseWord.find("email") != std::string::npos) usernameScore += 4.0; 
                if (lowercaseWord.find("mail") != std::string::npos) usernameScore += 3.0;  
                if (lowercaseWord.find("name") != std::string::npos) usernameScore += 2.0;  
                if (lowercaseWord.find("phone") != std::string::npos) usernameScore += 2.0; 
                if (lowercaseWord.find("account") != std::string::npos) usernameScore += 1.5; 
                if (lowercaseWord.find("id") != std::string::npos) usernameScore += 1.5;    

                // Password indicators with adjusted weights
                if (lowercaseWord.find("pass") != std::string::npos) passwordScore += 4.0;  
                if (lowercaseWord == "pw") passwordScore += 3.0;
                if (lowercaseWord.find("pin") != std::string::npos) passwordScore += 1.5;   
            }
        }

        // Position heuristic: improved for different login layouts
        if (i == 0) usernameScore += 1.5;  // Increased likelihood for first field
        if (i == 1 && inputFields.size() >= 2) passwordScore += 1.5;  // Increased likelihood for second field

        // Visual appearance heuristic - if field has password masking dots already visible
        int possibleDots = countPasswordDots(fieldRegion);
        if (possibleDots > 0) {
            passwordScore += 3.0 + std::min(possibleDots, 8) * 0.3;  // More dots = higher confidence
        }

        // Check field's relative position (password fields typically come after username fields)
        if (i > 0 && i < inputFields.size()) {
            // If this field is below a field with high username score, boost password score
            for (size_t j = 0; j < i; j++) {
                if (inputFields[j].y < field.y &&
                    std::abs((inputFields[j].x + inputFields[j].width / 2) - (field.x + field.width / 2)) < field.width) {
                    passwordScore += 1.0;
                    break;
                }
            }
        }

        // Record scores for this field
        if (usernameScore > 0) usernameScores.push_back({ i, usernameScore });
        if (passwordScore > 0) passwordScores.push_back({ i, passwordScore });
    }

    // Find the fields with highest scores
    int usernameFieldIdx = -1;
    int passwordFieldIdx = -1;
    double maxUsernameScore = 0.0;
    double maxPasswordScore = 0.0;

    for (const auto& pair : usernameScores) {
        if (pair.second > maxUsernameScore) {
            maxUsernameScore = pair.second;
            usernameFieldIdx = pair.first;
        }
    }

    for (const auto& pair : passwordScores) {
        if (pair.second > maxPasswordScore) {
            maxPasswordScore = pair.second;
            passwordFieldIdx = pair.first;
        }
    }

    // If same field has high scores for both, pick based on highest score with a bias toward password
    if (usernameFieldIdx == passwordFieldIdx && usernameFieldIdx != -1) {
        // Password detection is generally more reliable, so give it slight preference
        if (maxPasswordScore > maxUsernameScore * 0.9) {
            usernameFieldIdx = -1;
        }
        else {
            passwordFieldIdx = -1;
        }
    }

    // Fallback to position heuristics with improved logic
    if (usernameFieldIdx == -1 && passwordFieldIdx == -1) {
        if (inputFields.size() >= 2) {
            // Look for fields arranged vertically one after another
            for (size_t i = 0; i < inputFields.size() - 1; i++) {
                cv::Rect upper = inputFields[i];
                cv::Rect lower = inputFields[i + 1];

                // Check if fields are vertically aligned and close to each other
                if (lower.y > upper.y &&
                    lower.y - (upper.y + upper.height) < upper.height * 2 &&
                    std::abs((upper.x + upper.width / 2) - (lower.x + lower.width / 2)) < upper.width) {

                    usernameFieldIdx = i;
                    passwordFieldIdx = i + 1;
                    break;
                }
            }

            // If still not found, use simple first/second heuristic
            if (usernameFieldIdx == -1) {
                usernameFieldIdx = 0;
                passwordFieldIdx = 1;
            }
        }
        else if (inputFields.size() == 1) {
            // Only one field - more likely to be username
            usernameFieldIdx = 0;
        }
    }
    else if (usernameFieldIdx == -1 && passwordFieldIdx != -1) {
        // Password field found but no username field - look for a field above it
        for (size_t i = 0; i < inputFields.size(); i++) {
            if (i != passwordFieldIdx &&
                inputFields[i].y < inputFields[passwordFieldIdx].y &&
                std::abs((inputFields[i].x + inputFields[i].width / 2) -
                    (inputFields[passwordFieldIdx].x + inputFields[passwordFieldIdx].width / 2)) < inputFields[i].width) {
                usernameFieldIdx = i;
                break;
            }
        }

        // If still not found and password isn't the first field, assume the first is username
        if (usernameFieldIdx == -1 && passwordFieldIdx > 0) {
            usernameFieldIdx = 0;
        }
    }
    else if (passwordFieldIdx == -1 && usernameFieldIdx != -1) {
        // Username field found but no password field - look for a field below it
        for (size_t i = 0; i < inputFields.size(); i++) {
            if (i != usernameFieldIdx &&
                inputFields[i].y > inputFields[usernameFieldIdx].y &&
                std::abs((inputFields[i].x + inputFields[i].width / 2) -
                    (inputFields[usernameFieldIdx].x + inputFields[usernameFieldIdx].width / 2)) < inputFields[i].width) {
                passwordFieldIdx = i;
                break;
            }
        }

        // If still not found and username isn't the last field, check the next field
        if (passwordFieldIdx == -1 && usernameFieldIdx < inputFields.size() - 1) {
            passwordFieldIdx = usernameFieldIdx + 1;
        }
    }

    // Extract username content
    if (usernameFieldIdx != -1) {
        fields.usernameFieldPresent = true;
        cv::Mat usernameFieldImg = image(inputFields[usernameFieldIdx]);
        fields.username = extractUsernameContent(usernameFieldImg, words);
    }

    // Count password dots
    if (passwordFieldIdx != -1) {
        fields.passwordFieldPresent = true;
        cv::Mat passwordFieldImg = image(inputFields[passwordFieldIdx]);
        fields.passwordDots = countPasswordDots(passwordFieldImg);
    }

    return fields;
}

int LoginDetector::countPasswordDots(const cv::Mat& passwordField) {
    // Convert to grayscale for processing
    cv::Mat grayField;
    cv::cvtColor(passwordField, grayField, cv::COLOR_BGR2GRAY);

    // Apply adaptive thresholding for better dot detection - adjusted parameters
    cv::Mat binary;
    cv::adaptiveThreshold(grayField, binary, 255, cv::ADAPTIVE_THRESH_MEAN_C,
        cv::THRESH_BINARY_INV, 11, 5); // Changed from GAUSSIAN to MEAN, adjusted C value

    // Apply morphological operations to isolate dots - smaller kernel for smaller dots
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2)); // Reduced from 3x3
    cv::Mat morphed;
    cv::morphologyEx(binary, morphed, cv::MORPH_OPEN, kernel);

    // Multi-strategy approach for dot counting
    int dotCount = 0;

    // Strategy 1: Connected component analysis with adjusted parameters
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(morphed, labels, stats, centroids);

    std::vector<int> dotAreas;
    for (int i = 1; i < numComponents; i++) { // Skip background (label 0)
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // More lenient size filtering - allow smaller dots
        if (area > 1 && area < 400 && // Changed from 3-300 to 1-400
            std::abs(width - height) < width * 0.7 && // More lenient aspect ratio check
            width < passwordField.cols / 5) { // Wider dots allowed
            dotAreas.push_back(area);
        }
    }

    // If we found potential dots, use their count with improved filtering
    if (!dotAreas.empty()) {
        // Sort areas to find the most common dot size
        std::sort(dotAreas.begin(), dotAreas.end());

        // Find median dot area
        int medianArea = dotAreas[dotAreas.size() / 2];

        // Count dots with similar size to the median (more lenient filtering)
        int count = 0;
        for (int area : dotAreas) {
            if (area > medianArea * 0.3 && area < medianArea * 3.0) { // More lenient range
                count++;
            }
        }

        dotCount = count;
    }

    // Strategy 2: Improved Hough Circles detection
    if (dotCount < 3) { // Only use if connected components found too few dots
        std::vector<cv::Vec3f> circles;
        cv::HoughCircles(grayField, circles, cv::HOUGH_GRADIENT, 1,
            grayField.rows / 40, // Reduced min distance between circles
            30, 15, 1, grayField.rows / 8); // More lenient parameters

        // If Hough circles found more dots, use that count
        if (circles.size() > dotCount) {
            dotCount = circles.size();
        }
    }

    // Strategy 3: Improved clustering algorithm
    // Create horizontal histogram of dark pixels with lower threshold
    std::vector<int> horizontalHist(passwordField.cols, 0);
    for (int x = 0; x < passwordField.cols; x++) {
        for (int y = 0; y < passwordField.rows; y++) {
            if (binary.at<uchar>(y, x) > 0) {
                horizontalHist[x]++;
            }
        }
    }

    // Count clusters of dark pixels with more adaptive threshold
    bool inCluster = false;
    int clusters = 0;
    int minWidth = 1; // Reduced minimum width to catch smaller dots
    int currentWidth = 0;
    double threshold = passwordField.rows * 0.15; // Lower threshold (15% vs 20%)

    for (int x = 0; x < passwordField.cols; x++) {
        if (horizontalHist[x] > threshold) { // Lower threshold for "dark" column
            if (!inCluster) {
                inCluster = true;
                currentWidth = 1;
            }
            else {
                currentWidth++;
            }
        }
        else {
            if (inCluster) {
                if (currentWidth >= minWidth) {
                    clusters++;
                }
                inCluster = false;
            }
        }
    }

    // Check the last cluster
    if (inCluster && currentWidth >= minWidth) {
        clusters++;
    }

    // If cluster method found more dots, use that count
    if (clusters > dotCount && clusters < 30) { // Upper limit to avoid false positives
        dotCount = clusters;
    }

    // Strategy 4: Uniform pattern detection for evenly spaced dots
    if (dotCount < 3) { // Only use if other methods found few dots
        int uniformPatternCount = 0;
        std::vector<int> peakLocations;

        // Find peaks in horizontal histogram (potential dot centers)
        for (int x = 1; x < passwordField.cols - 1; x++) {
            if (horizontalHist[x] > horizontalHist[x - 1] &&
                horizontalHist[x] > horizontalHist[x + 1] &&
                horizontalHist[x] > threshold) {
                peakLocations.push_back(x);
            }
        }

        // Check for uniform spacing between peaks
        if (peakLocations.size() >= 3) {
            std::vector<int> distances;
            for (size_t i = 1; i < peakLocations.size(); i++) {
                distances.push_back(peakLocations[i] - peakLocations[i - 1]);
            }

            // Calculate average distance
            int sum = 0;
            for (int d : distances) sum += d;
            double avgDistance = static_cast<double>(sum) / distances.size();

            // Count peaks that follow the uniform pattern
            int validPeaks = 1; // First peak is always valid
            for (size_t i = 1; i < peakLocations.size(); i++) {
                double ratio = (peakLocations[i] - peakLocations[i - 1]) / avgDistance;
                if (ratio > 0.7 && ratio < 1.3) { // Within 30% of average
                    validPeaks++;
                }
            }

            uniformPatternCount = validPeaks;
        }

        if (uniformPatternCount > dotCount) {
            dotCount = uniformPatternCount;
        }
    }

    return dotCount;
}

std::string LoginDetector::extractUsernameContent(const cv::Mat& usernameField, const std::vector<WordBox>& words) {
    // Extract just the username field region
    cv::Rect fieldRect = cv::Rect(0, 0, usernameField.cols, usernameField.rows);

    // Find all words that intersect with the field with improved criteria
    std::string username;
    std::vector<WordBox> fieldWords;

    for (const auto& word : words) {
        // Check if the word is at least partially inside the field
        cv::Rect intersection = word.box & fieldRect;
        double overlapRatio = intersection.area() / static_cast<double>(word.box.area());

        // More lenient overlap requirement
        if (overlapRatio > 0.3 && word.confidence > 40) {
            fieldWords.push_back(word);
        }
    }

    // If OCR found words in the field, use them
    if (!fieldWords.empty()) {
        // Sort words by x-coordinate
        std::sort(fieldWords.begin(), fieldWords.end(),
            [](const WordBox& a, const WordBox& b) { return a.box.x < b.box.x; });

        // Combine words to form username
        for (const auto& word : fieldWords) {
            if (!username.empty()) {
                username += " ";
            }
            username += word.word;
        }
    }
    else {
        // If no words found via OCR, try a dedicated OCR just for this field with enhanced processing
        cv::Mat fieldImage = usernameField.clone();

        // Enhance contrast for better OCR - improved preprocessing
        cv::Mat enhanced;

        // Method 1: Normalize
        cv::normalize(fieldImage, enhanced, 0, 255, cv::NORM_MINMAX);

        // Method 2: Apply CLAHE for better local contrast
        cv::Mat gray;
        cv::cvtColor(enhanced, gray, cv::COLOR_BGR2GRAY);
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        cv::Mat claheOutput;
        clahe->apply(gray, claheOutput);

        // Method 3: Apply adaptive threshold for binary image
        cv::Mat binary;
        cv::adaptiveThreshold(claheOutput, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY, 11, 2);

        // Method 4: Invert if dark text on light background
        cv::Scalar meanIntensity = cv::mean(gray);
        if (meanIntensity[0] > 127) {
            cv::bitwise_not(binary, binary);
        }

        // Run Tesseract with optimized parameters
        tesseract::TessBaseAPI tess;
        tess.Init(NULL, "eng");

        // Try multiple image variants and use the best result
        std::vector<cv::Mat> variants = { claheOutput, binary };
        std::string bestText;
        int bestConfidence = 0;

        for (const auto& variant : variants) {
            tess.SetImage(variant.data, variant.cols, variant.rows,
                variant.channels(), variant.step);
            tess.SetSourceResolution(300);
            tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);

            // Get confidence of result
            tess.Recognize(NULL);
            int currentConfidence = tess.MeanTextConf();

            char* text = tess.GetUTF8Text();
            if (text != NULL) {
                std::string currentText = std::string(text);
                // Clean up the extracted text
                currentText.erase(std::remove(currentText.begin(), currentText.end(), '\n'), currentText.end());
                currentText.erase(std::remove(currentText.begin(), currentText.end(), '\r'), currentText.end());

                // Keep the highest confidence result
                if (currentConfidence > bestConfidence && !currentText.empty()) {
                    bestConfidence = currentConfidence;
                    bestText = currentText;
                }

                delete[] text;
            }
        }

        username = bestText;
        tess.End();
    }

    // Clean up the username text
    // Remove common placeholder texts
    std::vector<std::string> placeholders = {
        "username", "user name", "email", "email address", "phone", "login", "user id"
    };

    std::string lowerUsername = username;
    std::transform(lowerUsername.begin(), lowerUsername.end(), lowerUsername.begin(), ::tolower);

    for (const auto& placeholder : placeholders) {
        if (lowerUsername == placeholder) {
            return ""; // Don't return placeholder text
        }
    }

    return username;
}

// Calculates the confidence score that an image contains a login screen based on recognized text and UI analysis
float LoginDetector::computeLoginConfidence(const std::string& recognizedText,
    const std::vector<WordBox>& words,
    bool isDarkTheme) {
    // Start with zero base confidence
    float baseConfidence = 0.0f;

    // Check for strong keywords that are highly indicative of login screens
    // If found, set a high base confidence level (0.8)
    for (const auto& keyword : strongKeywords) {
        if (recognizedText.find(keyword) != std::string::npos) {
            Logger::log(Logger::Level::DEBUG, "Strong keyword found: " + keyword);
            baseConfidence = std::max(baseConfidence, 0.8f);
        }
    }

    // Analyze individual words with high OCR confidence
    // Accumulate confidence for each login-related term found
    float wordConfidence = 0.0f;
    for (const auto& word : words) {
        // Only consider words with OCR confidence > 60%
        if (word.confidence > 60) {
            for (const auto& keyword : loginKeywords) {
                // Match either exact keywords or keywords within longer words
                if (word.word == keyword || (word.word.length() > 4 && word.word.find(keyword) != std::string::npos)) {
                    wordConfidence += 0.1f;  // Increment confidence for each match
                    Logger::log(Logger::Level::DEBUG, "High confidence login word: " + word.word + " (" +
                        std::to_string(word.confidence) + "%)");
                    break;  // No need to check other keywords for this word
                }
            }
        }
    }

    // Cap word-based confidence at 0.7 to prevent overconfidence
    wordConfidence = std::min(wordConfidence, 0.7f);

    // Take the higher of base confidence or word confidence
    baseConfidence = std::max(baseConfidence, wordConfidence);

    // Begin feature-based confidence calculation
    float featureConfidence = 0.0f;

    // Check for specific login form features

    // 1. Check for username/email/phone field
    bool hasEmailField = recognizedText.find("email") != std::string::npos ||
        recognizedText.find("username") != std::string::npos ||
        recognizedText.find("phone") != std::string::npos;

    // 2. Check for password field
    bool hasPasswordField = recognizedText.find("password") != std::string::npos;

    // 3. Check for submit/login button
    bool hasSubmitButton = recognizedText.find("sign in") != std::string::npos ||
        recognizedText.find("log in") != std::string::npos ||
        recognizedText.find("login") != std::string::npos ||
        recognizedText.find("continue") != std::string::npos ||
        recognizedText.find("next") != std::string::npos;

    // 4. Check for account recovery or creation options
    bool hasAccountOptions = recognizedText.find("forgot") != std::string::npos ||
        recognizedText.find("create account") != std::string::npos ||
        recognizedText.find("sign up") != std::string::npos ||
        recognizedText.find("register") != std::string::npos;

    // 5. Check for alternative login methods (social, OAuth)
    bool hasAlternativeLogins = recognizedText.find("continue with") != std::string::npos ||
        recognizedText.find("sign in with") != std::string::npos ||
        (recognizedText.find("google") != std::string::npos &&
            recognizedText.find("facebook") != std::string::npos);

    // Add confidence based on critical form elements
    if (hasEmailField && hasPasswordField) featureConfidence += 0.4f;  // Both username and password fields
    else if (hasEmailField || hasPasswordField) featureConfidence += 0.2f;  // Either field alone

    // Add confidence for submit button
    if (hasSubmitButton) featureConfidence += 0.2f;

    // Add confidence for account options
    if (hasAccountOptions) featureConfidence += 0.1f;

    // Add confidence for alternative login methods
    if (hasAlternativeLogins) featureConfidence += 0.1f;

    // Small adjustment for dark themes, which are common in login interfaces
    float themeAdjustment = isDarkTheme ? 0.05f : 0.0f;

    // Combine the highest confidence method with theme adjustment
    float finalConfidence = std::max(baseConfidence, featureConfidence) + themeAdjustment;

    // Ensure confidence doesn't exceed 1.0
    finalConfidence = std::min(finalConfidence, 1.0f);

    // Log detailed confidence calculations for debugging
    Logger::log(Logger::Level::DEBUG, "Base confidence: " + std::to_string(baseConfidence) +
        ", Feature confidence: " + std::to_string(featureConfidence) +
        ", Theme adjustment: " + std::to_string(themeAdjustment) +
        ", Final confidence: " + std::to_string(finalConfidence));

    return finalConfidence;
}