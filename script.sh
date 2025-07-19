# header-begin ------------------------------------------
# File       : script.sh
#
# Author      : Joshua E
# Email       : estesjn2020@gmail.com
#
# Created on  : 7/18/2025
#
# header-end --------------------------------------------

#!/bin/bash

source /apps/profiles/modules_asax.sh.dyn

module load oneapi
module load cuda

cp ASA/Makefile .

make

LD_LIBRARY_PATH=/home/aubjne001/opencv_install/lib64:/apps/x86-64/apps/intel_2023.1.0/ipp/latest/lib/intel64:$LD_LIBRARY_PATH --config-file off --export /home/aubjne001/4k_profile --force-overwrite --set full --import-source yes main ./main

# footer-begin ------------------------------------------
# default.Shell
# File       : script.sh
# footer-end --------------------------------------------