#!/bin/bash
### 
# This file can be used to process a list of one or more experiments at different angles. 
# You should only need to change the parameters within the "Parameters to Change" block. 
# All other code shouldn't be changed unless more advanced functionality is needed. 
###

###################  Parameters to Change ###################
# For names/ids, these should be lists of the names/ids of the objects to process.
#       One of these two lists can be left empty, and it will be filled in accordingly
# For exp_num, angles, is_los_s, these should be:
#       - a list with the same length as names/ids (e.g., ("2" "2")) OR
#       - an empty list (e.g., ()) OR
#       - a single string which will be applied for all objects (e.g., "2")
# radar_type should be either "77_ghz" or "24_ghz"
names=("spatula") 
ids=("033") 
exp_num="1" 
angles=()
is_los_s="n"
radar_type="24_ghz"
##############################################################

num_names=${#names[@]}
num_ids=${#ids[@]}
num_angles=${#angles[@]}
num_is_los=${#is_los_s[@]}
num_exps=${#exp_num[@]}
# Check the name/id lists are valid
if [[ $num_names -ne $num_ids  && $num_names -ne 0 && $num_ids -ne 0 ]]; then
    echo "The names and ids list should either have the same length, or one should be empty"
    exit 1
fi

num_objs=$(( $num_names > $num_ids ? $num_names : $num_ids ))
# Check the angle list is valid
if [[ $num_angles -ne $num_objs && $num_angles -ne 0 && $num_angles -ne 1 ]]; then
    echo "angles should either be a list with the same length as ids/names, be a single string, or be an empty list."
    exit 1
fi

if [[ $num_is_los -ne $num_objs && $num_is_los -ne 0 && $num_is_los -ne 1 ]]; then
    echo "is_los_s should either be a list with the same length as ids/names, be a single string, or be an empty list."
    exit 1
fi

if [[ $num_exps -ne $num_objs && $num_exps -ne 0 && $num_exps -ne 1 ]]; then
    echo "exp_nums should either be a list with the same length as ids/names, be a single string, or be an empty list."
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

    if test $num_exps -eq 1; then
        num=$exp_num
    elif test $num_exps -ne 0; then
        num="${exp_num[i]}"
    else 
        num="1"
    fi

    if test $num_angles -eq 1; then
        angle=$angles
    elif test $num_angles -ne 0; then
        angle="${angles[i]}"
    else
        angle="None"
    fi

    if test $num_is_los -eq 1; then
        is_los=$is_los_s
    elif test $num_is_los -ne 0; then
        is_los="${is_los_s[i]}"
    else
        is_los= "y"
    fi

    python3 robot_imager.py --name $name --id $id --radar_type $radar_type --angles $angle --is_los $is_los --exp_num $num --ext "" 
done