# Face Expression Detector Setup Script
Write-Host "🚀 Face Expression Detector Setup" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green

Write-Host "`n📦 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`n🧠 Generating dummy model..." -ForegroundColor Yellow
python generate_dummy_model.py

Write-Host "`n🔍 Testing setup..." -ForegroundColor Yellow
python test_setup.py

Write-Host "`n✅ Setup complete! Run 'python emotion_detector.py' to start." -ForegroundColor Green
Read-Host "Press Enter to continue"
