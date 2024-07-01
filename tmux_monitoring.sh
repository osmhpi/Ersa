#!/bin/bash                                                                                                   

SESSIONNAME="tmux-monitor"
tmux has-session -t $SESSIONNAME &> /dev/null

if [ $? != 0 ] 
 then
    tmux new-session -s $SESSIONNAME -d
    tmux send-keys -t $SESSIONNAME 'watch nvidia-smi' C-m 
    tmux split-window -h
    tmux send-keys -t $SESSIONNAME 'htop' C-m
    tmux select-pane -t 0
    tmux split-window
    tmux send-keys -t $SESSIONNAME 'source /home/Felix.Grzelka/megaclite/venv/bin/activate' C-m 'watch python3 check_pids.py' C-m
fi

tmux attach -t $SESSIONNAME