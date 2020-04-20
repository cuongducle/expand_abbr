#!/bin/bash 

CDIR="`dirname "$0"`"
HOME_PATH=`(cd "$CDIR"/ ; pwd)`
FILE_MAIN=expand_api.py
LOG_PATH=$HOME_PATH/vbee.log
PID_NAME=$HOME_PATH/expand-abbr.pid

start()
{
	cd $HOME_PATH
	echo $HOME_PATH
	/home/kynh/anaconda3/envs/expand_abbr/bin/python $FILE_MAIN > $LOG_PATH  2>&1 &
		echo $! > $PID_NAME
	echo "$FILE_MAIN running with pid=`(cat $PID_NAME)`" 
}

stop()
{
	kill $( cat $PID_NAME)
	rm $PID_NAME
	echo "$FILE_MAIN stoped" 
}

case  "$1"  in 
	start)
	start	
	sleep 2		
	;;
	
	stop)
	stop
	;;

	restart)
	stop
	sleep 2
	start	
	;;
		
	*)
	start	
	;;
esac

