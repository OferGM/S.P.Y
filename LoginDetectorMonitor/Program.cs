using System;
using System.Threading;
using LoginDetectorMonitor.Configuration;
using LoginDetectorMonitor.Core;
using LoginDetectorMonitor.Logging;

namespace LoginDetectorMonitor
{
    class Program
    {
        private static LoginMonitor _monitor;
        private static readonly ManualResetEvent _exitEvent = new ManualResetEvent(false);

        static void Main(string[] args)
        {
            try
            {
                // Initialize configuration
                AppConfig config = AppConfig.LoadConfig();

                // Setup logging
                Logger.Initialize(config.LogFilePath);
                Logger.Log("Application started");

                Console.WriteLine($"Starting Login Detector Monitor...");
                Console.WriteLine($"Log file: {config.LogFilePath}");
                Console.WriteLine($"Monitoring duration: {config.MaxCaptureDuration} minutes");

                // Create and start the login monitor
                _monitor = new LoginMonitor(config);
                _monitor.Start();

                // Wait for exit signal
                _exitEvent.WaitOne();

                // Cleanup
                _monitor.Stop();
                Logger.Log("Application stopped");

                Console.WriteLine("Login Detector Monitor stopped successfully.");
            }
            catch (Exception ex)
            {
                Logger.Log($"Unhandled exception: {ex}");
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine("Press any key to exit...");
                Console.ReadKey();
            }
        }
    }
}
