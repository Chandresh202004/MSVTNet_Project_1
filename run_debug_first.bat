@echo off
echo Running MSVTNet Debug Checks for Subject 1
echo Current Time: %date% %time%
echo User: Chandresh202004

mkdir "D:\MSVTNet_Project\debug_results\session_dependent"
mkdir "D:\MSVTNet_Project\debug_results\session_independent"

echo Running session-dependent debug...
python debug_model.py --subject 1 --session_dependent --train_session 1 --test_session 2 --results_dir "D:\MSVTNet_Project\debug_results\session_dependent"

echo Running session-independent debug...
python debug_model.py --subject 1 --results_dir "D:\MSVTNet_Project\debug_results\session_independent"

echo Debug completed!
pause