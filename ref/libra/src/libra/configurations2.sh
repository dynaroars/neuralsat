#!/usr/bin/env bash
###################################################
# experiment A: different analysis configurations #
###################################################

LIBRA=$1

#=======#
# boxes #
#=======#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes-0.25-5.log
fi

#==========#
# symbolic #
#==========#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-symbolic-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain symbolic --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-symbolic-0.25-5.log
fi

#==========#
# deeppoly #
#==========#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly-0.25-5.log
fi

#==========#
# neurify #
#==========#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-neurify-0.25-5.log
fi

#================#
# boxes+deeppoly #
#================#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly-0.25-5.log
fi

#===============#
# boxes+neurify #
#===============#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_neurify --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes_neurify-0.25-5.log
fi

#===================#
# deeppoly+symbolic #
#===================#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_symbolic --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly_symbolic-0.25-5.log
fi

#==================#
# neurify+symbolic #
#==================#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain neurify_symbolic --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-neurify_symbolic-0.25-5.log
fi

#==================#
# deeppoly+neurify #
#==================#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify-0.25-5.log
fi

#========================#
# boxes+deeppoly+neurify #
#========================#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain boxes_deeppoly_neurify --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-boxes_deeppoly_neurify-0.25-5.log
fi

#===========================#
# deeppoly+neurify+symbolic #
#===========================#

if [ ! -z $2 ]
then
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.25 --upper 3 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.25 --upper 5 --cpu $2 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.25-5.log
else
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.5-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.5-5.log

    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.25 --upper 3 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.25-3.log
    $1 tests/japanese/japanese.txt tests/japanese/20.py --domain deeppoly_neurify_symbolic --lower 0.25 --upper 5 | tee tests/japanese/logs2/japanese-20-deeppoly_neurify_symbolic-0.25-5.log
fi
