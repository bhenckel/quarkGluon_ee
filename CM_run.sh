# script to run the python program, save the output to a log file and save the log file. 

pythonfile=$1
date_str="$(date '+%Y.%m.%d_%H.%M.%S')"
log_str="${date_str}_log.log"
python_str="${date_str}_${pythonfile}"

cp ${pythonfile} "CM_logs/${python_str}"

echo $pythonfile

python -u ${pythonfile} 2>&1 |  tee "CM_logs/${log_str}"  & disown

rm CM_out.log
ln -s "CM_logs/${log_str}" CM_out.log
