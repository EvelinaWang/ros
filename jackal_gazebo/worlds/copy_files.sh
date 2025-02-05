#!/bin/bash

SOURCE_DIR="/home/evelina/the-barn-challenge/jackal_helper/worlds/BARN"
DEST_DIR="$HOME/catkin_ws/src/jackal_simulator/jackal_gazebo/worlds"

# Iterate through each file in the source directory
for file in "$SOURCE_DIR"/*; do
    # Copy the file to the destination directory
    cp "$file" "$DEST_DIR"
    echo "Copied $(basename "$file") to $DEST_DIR"
done
