@echo off
echo Starting MSVTNet training for all subjects...
echo Current Time: %date% %time%
echo User: Chandresh202004

REM Create necessary directories
mkdir "D:\MSVTNet_Project\models\msvtnet\session_dependent"
mkdir "D:\MSVTNet_Project\models\msvtnet\session_independent"
mkdir "D:\MSVTNet_Project\results\msvtnet\session_dependent"
mkdir "D:\MSVTNet_Project\results\msvtnet\session_independent"

REM Define common parameters
set COMMON_PARAMS=--mixed_precision --epochs 150 --batch_size 32 --lr 0.001 --dropout 0.5 --augment --early_stopping 30 --scheduler cosine --num_workers 0
set SD_SAVE=--save_dir "D:\MSVTNet_Project\models\msvtnet\session_dependent" --results_dir "D:\MSVTNet_Project\results\msvtnet\session_dependent"
set SI_SAVE=--save_dir "D:\MSVTNet_Project\models\msvtnet\session_independent" --results_dir "D:\MSVTNet_Project\results\msvtnet\session_independent"

REM Session-Dependent Training
echo =========================================
echo Starting Session-Dependent Training
echo =========================================

echo Starting Subject 1 (Session-Dependent)
python train.py --subject 1 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 2 (Session-Dependent)
python train.py --subject 2 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 3 (Session-Dependent)
python train.py --subject 3 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 4 (Session-Dependent)
python train.py --subject 4 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 5 (Session-Dependent)
python train.py --subject 5 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 6 (Session-Dependent)
python train.py --subject 6 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 7 (Session-Dependent)
python train.py --subject 7 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 8 (Session-Dependent)
python train.py --subject 8 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

echo Starting Subject 9 (Session-Dependent)
python train.py --subject 9 --session_dependent --train_session 1 --test_session 2 %COMMON_PARAMS% %SD_SAVE%

REM Session-Independent Training
echo =========================================
echo Starting Session-Independent Training
echo =========================================

echo Starting Subject 1 (Session-Independent)
python train.py --subject 1 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 2 (Session-Independent)
python train.py --subject 2 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 3 (Session-Independent)
python train.py --subject 3 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 4 (Session-Independent)
python train.py --subject 4 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 5 (Session-Independent)
python train.py --subject 5 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 6 (Session-Independent)
python train.py --subject 6 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 7 (Session-Independent)
python train.py --subject 7 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 8 (Session-Independent)
python train.py --subject 8 %COMMON_PARAMS% %SI_SAVE%

echo Starting Subject 9 (Session-Independent)
python train.py --subject 9 %COMMON_PARAMS% %SI_SAVE%

echo All training completed!
echo Final completion time: %date% %time%
pause