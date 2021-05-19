#!/bin/bash
#---------------------------------------------------------------------#
input='baselines_run.txt'
#---------------------------------------------------------------------#
while true
do
  # for each line in the baselines run file we check whether it is running or not
  # if it is not running we launch it
  echo "Reading" $input
  while IFS= read -r line
  do
    line="$(echo $line| sed -n -e 's/[\r\n\t]*$//p'| sed -n -e 's/^[ \r\n\t]*//p')"
    command_text="$(echo $line |sed -n -e 's/^.*python main.py \(.*\)/\1/p'| sed -n -e 's/[\r\n \t]*$//p')"
    lines="$(bbjobs|sed -n -e "s/.*\($command_text\).*/\1/p")"
    num_lines="$(bbjobs|sed -n -e "s/.*\($command_text.*\)/\1/p"|wc -l)"
    if [[ $num_lines -lt 1 ]]
    then
      # launch the job
      echo $line
      echo ""

    fi
  done < $input
  echo "Going back to sleep"
  sleep 10000 # check again in 1000 seconds
done
#---------------------------------------------------------------------#