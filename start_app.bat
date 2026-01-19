@echo off
TITLE A-VITAL SYSTEM LAUNCHER
CLS

echo ==================================================
echo   A-VITAL BIOLOGICAL INTELLIGENCE NODE
echo   Starting System...
echo ==================================================

echo.
echo [1/4] Installing Neural Dependencies...
pip install flask flask-cors pandas numpy scikit-learn joblib --quiet
echo    ^> Dependencies installed.

echo.
echo [2/4] Initializing Neural Core (Training Model)...
python train_model.py
echo    ^> Core initialized.

echo.
echo [3/4] Launching Server Node...
echo    ^> Opening separate server window...
start "A-VITAL SERVER" cmd /k "python app.py"

echo.
echo [4/4] Connecting Interface...
echo    ^> Waiting for server handshake (5s)...
timeout /t 5 /nobreak >nul

echo.
echo    ^> LAUNCHING DASHBOARD...
start http://localhost:5000

echo.
echo ==================================================
echo   SYSTEM ONLINE
echo   Do not close the "A-VITAL SERVER" window.
echo ==================================================
pause
exit
