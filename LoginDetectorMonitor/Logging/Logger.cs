using System;
using System.IO;

namespace LoginDetectorMonitor.Logging
{
    public static class Logger
    {
        private static string _logFilePath;

        public static void Initialize(string logFilePath)
        {
            _logFilePath = logFilePath;
            try
            {
                File.WriteAllText(_logFilePath, $"Log started at {DateTime.Now}\n");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Logger initialization error: {ex.Message}");
            }
        }

        public static void Log(string message)
        {
            if (string.IsNullOrEmpty(_logFilePath)) return;
            try
            {
                File.AppendAllText(_logFilePath, $"{DateTime.Now:HH:mm:ss.fff}: {message}\n");
            }
            catch { }
        }
    }
}