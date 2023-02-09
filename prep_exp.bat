@title MAIN WINDOW

:: PATHS
@set py_path="C:\Users\chick\AppData\Local\Programs\Python\Python39\python.exe"
@set exp_script_path="c:\wut\asura\Motor-Execution-Classifiaction\exps\psycho_er_exp.py"
@set pid_script_path="c:\wut\asura\Motor-Execution-Classifiaction\utils\gen_participant_id.py"

@set lsl_labrec_path="c:\wut\asura\labstreaminglayer\Apps\LabRecorder\LabRecorder.exe"
@set lsl_labrec_cfg_path="c:\wut\asura\labstreaminglayer\Apps\LabRecorder\LabRecorder.cfg"

@set lsl_unicorn_path="C:\Users\chick\OneDrive\Documents\gtec\Unicorn Suite\Hybrid Black\Unicorn LSL\UnicornLSL.exe"
@set lsl_gamepad_dir_path="c:\wut\asura\labstreaminglayer\Apps\App-Gamepad"


:: SURVEY
%py_path% %pid_script_path%

@echo #######################################
@echo Fill Experiments table then press Enter
@echo #######################################
@pause

@echo .......................................

@echo #######################################
@echo Connect devices then press Enter
@echo #######################################
@pause

:: RUN DEVICE LINK LSL PROC
start "" %lsl_unicorn_path%
start "" "%lsl_gamepad_dir_path%\GamepadLSL.exe" "%lsl_gamepad_dir_path%\GamepadLSL.cfg"
start "Experiment" cmd /C "%py_path% %exp_script_path%"

@echo #######################################
@echo Link all the devices then press Enter
@echo #######################################
@pause

:: RUN REMAINING PROC
start "LabRecorder" cmd /C "%lsl_labrec_path% -c %lsl_labrec_cfg_path%"
start "EEG Viewer" cmd /C "%py_path% .\utils\eeg_viewer.py"

@echo #######################################
@echo Press Enter when session ended
@echo #######################################
@pause
