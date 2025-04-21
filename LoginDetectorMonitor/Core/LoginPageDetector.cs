using System;
using System.Diagnostics;
using System.Threading;
using LoginDetectorMonitor.Configuration;
using LoginDetectorMonitor.Logging;

namespace LoginDetectorMonitor.Core
{
    public class LoginPageDetector
    {
        private readonly AppConfig _config;
        private readonly object _detectionLock = new object();
        private Stopwatch _processingTimeWatch = new Stopwatch();
        private int _consecutiveFailures = 0;
        private int _adaptiveDelayMs = 0;

        public LoginPageDetector(AppConfig config)
        {
            _config = config;
        }

        public bool IsLoginPage(string screenshotPath)
        {
            lock (_detectionLock) // Ensure only one detection process runs at a time
            {
                try
                {
                    // Track processing time for adaptive scaling
                    _processingTimeWatch.Restart();

                    // Apply adaptive delay if we've had consecutive failures
                    if (_adaptiveDelayMs > 0)
                    {
                        Thread.Sleep(_adaptiveDelayMs);
                    }

                    // Call the external detector process
                    using (Process process = new Process())
                    {
                        process.StartInfo = new ProcessStartInfo
                        {
                            FileName = _config.LoginDetectorPath,
                            Arguments = $"\"{screenshotPath}\"",
                            RedirectStandardOutput = true,
                            UseShellExecute = false,
                            CreateNoWindow = true
                        };
                        process.Start();

                        // Add timeout to prevent hanging
                        if (!process.WaitForExit(5000)) // 5 second timeout
                        {
                            Logger.Log("LoginDetector process timed out, killing process");
                            try { process.Kill(); } catch { }
                            _consecutiveFailures++;
                            AdjustAdaptiveDelay();
                            return false;
                        }

                        string output = process.StandardOutput.ReadToEnd();

                        // Reset failure counter on success
                        _consecutiveFailures = 0;
                        _adaptiveDelayMs = 0;

                        _processingTimeWatch.Stop();
                        return output.Trim().EndsWith("true", StringComparison.OrdinalIgnoreCase);
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log($"Login detection error: {ex.Message}");

                    _consecutiveFailures++;
                    AdjustAdaptiveDelay();
                    return false;
                }
                finally
                {
                    _processingTimeWatch.Stop();
                }
            }
        }

        // Adjust delay based on consecutive failures to prevent CPU thrashing
        private void AdjustAdaptiveDelay()
        {
            if (_consecutiveFailures > 5)
            {
                // Exponential backoff with maximum of 5 seconds
                _adaptiveDelayMs = Math.Min(5000, 100 * (int)Math.Pow(2, _consecutiveFailures - 5));
                Logger.Log($"Adding adaptive delay of {_adaptiveDelayMs}ms after {_consecutiveFailures} consecutive failures");
            }
        }
    }
}