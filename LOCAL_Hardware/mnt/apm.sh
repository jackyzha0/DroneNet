#!/bin/bash

#This script is launched automatically in Erle Robotics products
#on every boot and loads the autopilot

APM_BIN_DIR="/home/erle/PXFmini"
wifi="10.42.0.74:8080"
ros=`sudo systemctl is-enabled ros.service`
if [ "$ros" == "enabled" ]; then
    `sudo systemctl stop ros.service`
    `sudo systemctl disable ros.service`
    `sudo systemctl stop robot_blockly.service`
    `sudo systemctl disable robot_blockly.service`
    `sudo systemctl daemon-reload`
fi

GPS="/dev/ttyAMA0"
FLAGS="-l /home/erle/APM/logs"

while :; do
        $APM_BIN_DIR/ArduCopter.elf -A tcp:$wifi -B $GPS $FLAGS
done >> /home/erle/APM/info/copter.log 2>&1
