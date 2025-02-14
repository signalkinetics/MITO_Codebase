#!/bin/bash
### 
# This file can be used to simulate a list of one or more objects at different angles. 
# You should only need to change the parameters within the "Parameters to Change" block. 
# All other code shouldn't be changed unless more advanced functionality is needed. 
###

###################  Parameters to Change ###################
names=("spatula") 
ids=("None") 
angles=("0,0,0")
radar_type="77_ghz"
##############################################################

num_names=${#names[@]}
num_ids=${#ids[@]}
num_angles=${#angles[@]}
# Check the name/id lists are valid
if [[ $num_names -ne $num_ids  && $num_names -ne 0 && $num_ids -ne 0 ]]; then
    echo "The names and ids list should either have the same length, or one should be empty"
    exit 1
fi

num_objs=$(( $num_names > $num_ids ? $num_names : $num_ids ))
# Check the angle list is valid
if [[ $num_angles -ne $num_objs && $num_angles -ne 0 ]]; then
    echo "The angles list should either have the same length as ids/names or be empty."
    exit 1
fi

# Iterate over the list and run the Python script with each argument
for ((i=0; i<$num_objs; i++)); do
    if test $i -lt $num_names; then
        name="${names[i]}"
    else 
        name="None"
    fi
    if test $i -lt $num_ids; then
        id="${ids[i]}"
    else 
        id="None"
    fi

    if test $num_angles -ne 0; then
        angle="${angles[i]}"
    else
        angle="0,0,0"
    fi

    python3 run_simulation.py --name $name --id $id --radar_type $radar_type --angles $angle --ext "_sim_test"
done