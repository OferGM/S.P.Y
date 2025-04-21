//main.cpp
#include "login_detector.h"
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    // Validate command-line arguments
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <operation_mode> <path_to_screenshot>" << std::endl;
        std::cerr << "  operation_mode: 1 - Detect login screen, 2 - Extract fields" << std::endl;
        return 1;
    }

    // Parse operation mode
    int mode = std::stoi(argv[1]);
    if (mode != 1 && mode != 2) {
        std::cerr << "Invalid operation mode. Use 1 for login detection or 2 for field extraction." << std::endl;
        return 1;
    }

    // Start timing the detection process
    auto start = std::chrono::high_resolution_clock::now();

    // Initialize and configure the login detector
    LoginDetector detector;
    detector.setConfidenceThreshold(0.35f);

    if (mode == 1) {
        // Process the image and detect login screen
        bool isLoginScreen = detector.detectLogin(argv[2], LoginDetector::OperationMode::DETECT_LOGIN);

        // Calculate and display processing time
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Output results to standard output
        std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
        std::cout << "Login screen detected: " << (isLoginScreen ? "true" : "false") << std::endl;
    }
    else {
        // Extract username and password fields
        auto extractedFields = detector.extractLoginFields(argv[2]);

        // Calculate and display processing time
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        // Output results to standard output
        std::cout << "Processing time: " << duration.count() << " ms" << std::endl;
        std::cout << "Username field present: " << (extractedFields.usernameFieldPresent ? "true" : "false") << std::endl;
        std::cout << "Username content: " << extractedFields.username << std::endl;
        std::cout << "Password field present: " << (extractedFields.passwordFieldPresent ? "true" : "false") << std::endl;
        std::cout << "Password dots count: " << extractedFields.passwordDots << std::endl;
    }

    return 0;
}