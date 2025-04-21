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

    // Validate image file - quick early return if invalid
    if (!ImageUtils::isValidImageFile(imagePath)) return fields;

    // Load image and downscale large images for faster processing
    cv::Mat originalImage = cv::imread(imagePath);

    // Downscale large images
    if (originalImage.cols > 1200 || originalImage.rows > 1200) {
        float scale = 1200.0f / std::max(originalImage.cols, originalImage.rows);
        cv::resize(originalImage, originalImage, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // Detect theme - cheaper version
    bool isDarkTheme = ImageUtils::detectTheme(originalImage);

    // Skip full login detection - go directly to field detection
    // This saves significant time

    // Detect input fields first - this is faster than OCR
    std::vector<cv::Rect> inputFields = uiDetector.detectInputFields(originalImage, isDarkTheme);

    // Only perform OCR if we found input fields
    std::vector<WordBox> wordBoxes;
    if (!inputFields.empty()) {
        // Run OCR - use a simplified version when possible
        auto ocrResult = ocrProcessor.processImage(originalImage, isDarkTheme);
        wordBoxes = ocrResult.second;
    }
    else {
        return fields; // No input fields found, return early
    }

    // Analyze fields
    fields = analyzeLoginFields(originalImage, inputFields, wordBoxes);

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
                if (lowercaseWord.find("user") != std::string::npos) usernameScore += 4.0;  // Increased
                if (lowercaseWord.find("email") != std::string::npos) usernameScore += 4.0; // Increased
                if (lowercaseWord.find("mail") != std::string::npos) usernameScore += 3.0;  // Increased
                if (lowercaseWord.find("login") != std::string::npos) usernameScore += 2.0;
                if (lowercaseWord.find("name") != std::string::npos) usernameScore += 2.0;  // Increased
                if (lowercaseWord.find("phone") != std::string::npos) usernameScore += 2.0; // Increased
                if (lowercaseWord.find("account") != std::string::npos) usernameScore += 1.5; // Increased
                if (lowercaseWord.find("id") != std::string::npos) usernameScore += 1.5;    // Increased
                if (lowercaseWord.find("log") != std::string::npos) usernameScore += 1.0;   // Added
                if (lowercaseWord.find("sign") != std::string::npos) usernameScore += 1.0;  // Added

                // Password indicators with adjusted weights
                if (lowercaseWord.find("pass") != std::string::npos) passwordScore += 4.0;  // Increased
                if (lowercaseWord == "pw") passwordScore += 3.0;
                if (lowercaseWord.find("secret") != std::string::npos) passwordScore += 1.5; // Increased
                if (lowercaseWord.find("pin") != std::string::npos) passwordScore += 1.5;   // Increased
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

    // Fast blur to reduce noise while preserving dot shapes
    cv::Mat blurred;
    cv::medianBlur(grayField, blurred, 3);

    // Use binary thresholding - more reliable than adaptive for small dots
    cv::Mat binary;
    // Different threshold strategies for different themes
    double meanIntensity = cv::mean(blurred)[0];
    bool isDarkTheme = meanIntensity < 128;

    int threshValue = isDarkTheme ? 80 : 180;
    cv::threshold(blurred, binary, threshValue, 255, isDarkTheme ? cv::THRESH_BINARY : cv::THRESH_BINARY_INV);

    // Simple connected components analysis - fastest method
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(binary, labels, stats, centroids);

    // Filter components by size and shape
    std::vector<double> dotAreas;
    std::vector<cv::Point> dotCenters;

    for (int i = 1; i < numComponents; i++) { // Skip background (label 0)
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Check if this component has dot-like properties
        if (area >= 1 && area <= 150 &&
            width <= 20 && height <= 20 &&
            std::abs(width - height) <= 5) { // Roughly square/circular

            dotAreas.push_back(area);
            dotCenters.push_back(cv::Point(centroids.at<double>(i, 0), centroids.at<double>(i, 1)));
        }
    }

    // No dots found with primary method
    if (dotAreas.empty()) {
        return 0;
    }

    // Sort by area
    std::sort(dotAreas.begin(), dotAreas.end());

    // Determine the most common dot size
    double medianArea = dotAreas[dotAreas.size() / 2];

    // Extract dots with similar area to filter outliers
    std::vector<cv::Point> filteredDots;
    for (size_t i = 0; i < dotAreas.size(); i++) {
        if (dotAreas[i] >= medianArea * 0.3 && dotAreas[i] <= medianArea * 3.0) {
            filteredDots.push_back(dotCenters[i]);
        }
    }

    // Sort dots by horizontal position
    std::sort(filteredDots.begin(), filteredDots.end(),
        [](const cv::Point& a, const cv::Point& b) { return a.x < b.x; });

    // Get horizontal spacing stats for uniform pattern detection
    std::vector<int> spacings;
    for (size_t i = 1; i < filteredDots.size(); i++) {
        int spacing = filteredDots[i].x - filteredDots[i - 1].x;
        if (spacing > 0) {
            spacings.push_back(spacing);
        }
    }

    // No valid spacings found
    if (spacings.empty()) {
        return filteredDots.size();
    }

    // Calculate median spacing
    std::sort(spacings.begin(), spacings.end());
    int medianSpacing = spacings[spacings.size() / 2];

    // Count dots that form a uniform pattern with consistent spacing
    int patternCount = 1; // First dot is always part of pattern
    int startX = filteredDots[0].x;

    // Predict dot positions based on typical password masking patterns
    int predictedDots = 1;
    int fieldWidth = passwordField.cols;

    // If we have enough dots to establish a pattern
    if (filteredDots.size() >= 3) {
        predictedDots = (fieldWidth - 10) / medianSpacing;

        // Limit to reasonable range for password fields
        if (predictedDots > 20) predictedDots = 20;
        if (predictedDots < filteredDots.size()) predictedDots = filteredDots.size();
    }

    // Return the best count estimate - either actual dots or predicted pattern
    return predictedDots;
}

std::string LoginDetector::extractUsernameContent(const cv::Mat& usernameField, const std::vector<WordBox>& words) {
    // Extract from the username field region
    cv::Rect fieldRect = cv::Rect(0, 0, usernameField.cols, usernameField.rows);

    // Find all words that intersect with the field
    std::string username;
    std::vector<WordBox> fieldWords;

    // First pass - find words that are definitely inside the field
    for (const auto& word : words) {
        cv::Rect intersection = word.box & fieldRect;
        double overlapRatio = intersection.area() / static_cast<double>(word.box.area());

        if (overlapRatio > 0.6 && word.confidence > 60) {
            fieldWords.push_back(word);
        }
    }

    // Filter out placeholder text and button labels
    std::unordered_set<std::string> placeholderTexts = {
        "email", "email address", "phone", "username", "user name",
        "password", "sign in", "sign-in", "signin", "log in", "login",
        "use a sign-in code", "sign-in code", "code", "enter code"
    };

    std::vector<WordBox> contentWords;
    for (const auto& word : fieldWords) {
        std::string lowerWord = word.word;
        std::transform(lowerWord.begin(), lowerWord.end(), lowerWord.begin(), ::tolower);

        bool isPlaceholder = false;
        for (const auto& placeholder : placeholderTexts) {
            if (lowerWord == placeholder ||
                (placeholder.length() > 3 && lowerWord.find(placeholder) != std::string::npos)) {
                isPlaceholder = true;
                break;
            }
        }

        if (!isPlaceholder) {
            contentWords.push_back(word);
        }
    }

    // If we found content words, use them
    if (!contentWords.empty()) {
        // Sort by position
        std::sort(contentWords.begin(), contentWords.end(),
            [](const WordBox& a, const WordBox& b) { return a.box.x < b.box.x; });

        for (const auto& word : contentWords) {
            if (!username.empty()) username += " ";
            username += word.word;
        }
        return username;
    }

    // Quick visual check for empty fields
    cv::Mat fieldGray;
    cv::cvtColor(usernameField, fieldGray, cv::COLOR_BGR2GRAY);
    cv::Scalar meanIntensity = cv::mean(fieldGray);
    bool likelyEmpty = (meanIntensity[0] > 220 || meanIntensity[0] < 30);

    if (likelyEmpty) {
        return ""; // Field is likely empty
    }

    // Perform a targeted OCR on the field with enhanced preprocessing
    // This is a fallback for when our primary detection fails
    cv::Mat enhancedField;

    // Apply CLAHE for better local contrast
    cv::Mat gray;
    cv::cvtColor(usernameField, gray, cv::COLOR_BGR2GRAY);
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, enhancedField);

    // Simple thresholding for improved text extraction
    cv::Mat binary;
    cv::threshold(enhancedField, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // Run a targeted Tesseract OCR session
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng");
    tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    tess.SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@._-");

    tess.SetImage(binary.data, binary.cols, binary.rows, 1, binary.step);

    char* text = tess.GetUTF8Text();
    if (text != NULL) {
        username = std::string(text);
        // Clean up the text
        username.erase(std::remove(username.begin(), username.end(), '\n'), username.end());
        username.erase(std::remove(username.begin(), username.end(), '\r'), username.end());

        // Convert to lowercase for filtering
        std::string lowerUsername = username;
        std::transform(lowerUsername.begin(), lowerUsername.end(), lowerUsername.begin(), ::tolower);

        // Final check to filter out any remaining placeholders
        for (const auto& placeholder : placeholderTexts) {
            if (lowerUsername == placeholder) {
                username = "";
                break;
            }
        }

        delete[] text;
    }

    tess.End();
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