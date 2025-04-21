using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using LoginDetectorMonitor.Configuration;
using LoginDetectorMonitor.Logging;

namespace LoginDetectorMonitor.Core
{
    public class ScreenCapture
    {
        private readonly AppConfig _config;
        private readonly object _captureLock = new object();

        public ScreenCapture(AppConfig config)
        {
            _config = config;
        }

        public string CaptureScreen()
        {
            string screenshotPath = string.Empty;

            lock (_captureLock) // Ensure thread safety for GDI+ operations
            {
                try
                {
                    // Create a unique filename
                    screenshotPath = Path.Combine(
                        _config.ScreenshotDirectory,
                        $"screenshot_{DateTime.Now:yyyyMMdd_HHmmss_fff}.png");

                    // Capture the screenshot with specified dimensions
                    using (Bitmap screenshot = new Bitmap(_config.ScreenshotWidth, _config.ScreenshotHeight))
                    {
                        using (Graphics g = Graphics.FromImage(screenshot))
                        {
                            // Improve performance with these settings
                            g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighSpeed;
                            g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Low;
                            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighSpeed;

                            // Capture screen area
                            g.CopyFromScreen(0, _config.ScreenshotStartY, 0, 0, screenshot.Size);
                        }

                        // Use a more efficient saving mechanism
                        using (FileStream fs = new FileStream(screenshotPath, FileMode.Create, FileAccess.Write))
                        {
                            EncoderParameters encoderParams = new EncoderParameters(1);
                            encoderParams.Param[0] = new EncoderParameter(Encoder.Quality, 80L); // Reduce quality for performance

                            ImageCodecInfo jpegEncoder = GetEncoder(ImageFormat.Jpeg);
                            if (jpegEncoder != null)
                            {
                                // Use JPEG instead of PNG for better performance
                                screenshot.Save(fs, jpegEncoder, encoderParams);
                                screenshotPath = Path.ChangeExtension(screenshotPath, ".png");
                            }
                            else
                            {
                                // Fall back to PNG if JPEG encoder not available
                                screenshot.Save(fs, ImageFormat.Png);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Logger.Log($"Screenshot capture error: {ex.Message}");
                    screenshotPath = string.Empty;
                }
            }
            return screenshotPath;
        }

        private ImageCodecInfo GetEncoder(ImageFormat format)
        {
            ImageCodecInfo[] codecs = ImageCodecInfo.GetImageEncoders();

            foreach (ImageCodecInfo codec in codecs)
            {
                if (codec.FormatID == format.Guid)
                {
                    return codec;
                }
            }

            return null;
        }
    }
}