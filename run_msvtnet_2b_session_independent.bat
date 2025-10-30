@echo off
echo Starting MSVTNet Training for BCIC IV 2b (Session-Independent Approach)
echo Current Time: %date% %time%
echo User: Chandresh202004

REM Create necessary directories
mkdir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" 2>nul
mkdir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b" 2>nul

REM Define common parameters
set COMMON_PARAMS=--mixed_precision --epochs 150 --batch_size 16 --lr 0.0005 --dropout 0.3 --augment --weight_decay 0.0001 --early_stopping 50 --scheduler cosine --num_workers 0

REM Session-Independent Training
echo =========================================
echo Starting Session-Independent Training for BCIC IV 2b
echo =========================================

echo Starting Subject 1 (Session-Independent)
echo Time: %time%
python train.py --subject 1 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 1 (Session-Independent)
echo Time: %time%

echo Starting Subject 2 (Session-Independent)
echo Time: %time%
python train.py --subject 2 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 2 (Session-Independent)
echo Time: %time%

echo Starting Subject 3 (Session-Independent)
echo Time: %time%
python train.py --subject 3 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 3 (Session-Independent)
echo Time: %time%

echo Starting Subject 4 (Session-Independent)
echo Time: %time%
python train.py --subject 4 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 4 (Session-Independent)
echo Time: %time%

echo Starting Subject 5 (Session-Independent)
echo Time: %time%
python train.py --subject 5 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 5 (Session-Independent)
echo Time: %time%

echo Starting Subject 6 (Session-Independent)
echo Time: %time%
python train.py --subject 6 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 6 (Session-Independent)
echo Time: %time%

echo Starting Subject 7 (Session-Independent)
echo Time: %time%
python train.py --subject 7 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 7 (Session-Independent)
echo Time: %time%

echo Starting Subject 8 (Session-Independent)
echo Time: %time%
python train.py --subject 8 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 8 (Session-Independent)
echo Time: %time%

echo Starting Subject 9 (Session-Independent)
echo Time: %time%
python train.py --subject 9 --dataset_type 2a %COMMON_PARAMS% --progressive_training --data_dir "D:\MSVTNet_Project\datasets\BCIC_IV_2b\preprocessed" --save_dir "D:\MSVTNet_Project\models_2b\msvtnet_2b\session_independent_2b" --results_dir "D:\MSVTNet_Project\results_2b\msvtnet_2b\session_independent_2b"
echo Completed Subject 9 (Session-Independent)
echo Time: %time%

echo All session-independent training completed for BCIC IV 2b!
echo Final completion time: %date% %time%
pause