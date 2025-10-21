@echo off
echo 🚀 Face Expression Detector Setup
echo ================================

echo.
echo 📦 Installing dependencies...
pip install -r requirements.txt

echo.
echo 🧠 Generating dummy model...
python generate_dummy_model.py

echo.
echo 🔍 Testing setup...
python test_setup.py

echo.
echo ✅ Setup complete! Run 'python emotion_detector.py' to start.
pause
