#!/usr/bin/bash
  
echo "creating a temporary directory and some files"

source /storage/brno2/home/yauheni/miniconda3/bin/activate monai

echo "HI"


PYTHON_SCRIPT="/storage/brno2/home/yauheni/unetr3D/unetr_server.py"
LOG_FILE="/storage/brno2/home/yauheni/unetr3D/output_unetr3D_pigs.log"

echo "Python script: $PYTHON_SCRIPT"
echo "Log file: $LOG_FILE"



if [ ! -f "$LOG_FILE" ]; then
  touch "$LOG_FILE"
  echo "Log file created: $LOG_FILE"
else
  > "$LOG_FILE"
  echo "Log file already exists. Cleared contents of: $LOG_FILE"
fi


echo "Running Python script"
python $PYTHON_SCRIPT > $LOG_FILE 2>&1
echo "Python script finished"



# Проверяем статус выполнения Python скрипта
if [ $? -ne 0 ]; then
  echo "Python script encountered an error. Check $LOG_FILE for details."
else
  echo "Python script completed successfully. Check $LOG_FILE for details."
fi