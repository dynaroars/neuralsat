#!/usr/bin/env bash
############################
# experiment D: autotuning #
############################

LIBRA=$1

#=======#
# boxes #
#=======#

$1 tests/census/census.txt tests/census/20.py --domain boxes --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-boxes.log

#==========#
# symbolic #
#==========#

$1 tests/census/census.txt tests/census/20.py --domain symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-symbolic.log

#==========#
# deeppoly #
#==========#

$1 tests/census/census.txt tests/census/20.py --domain deeppoly --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-deeppoly.log

#==========#
# neurify #
#==========#

$1 tests/census/census.txt tests/census/20.py --domain neurify --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-neurify.log

#================#
# boxes+deeppoly #
#================#

$1 tests/census/census.txt tests/census/20.py --domain boxes_deeppoly --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-boxes_deeppoly.log

#===============#
# boxes+neurify #
#===============#

$1 tests/census/census.txt tests/census/20.py --domain boxes_neurify --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-boxes_neurify.log

#===================#
# deeppoly+symbolic #
#===================#

$1 tests/census/census.txt tests/census/20.py --domain deeppoly_symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-deeppoly_symbolic.log

#==================#
# neurify+symbolic #
#==================#

$1 tests/census/census.txt tests/census/20.py --domain neurify_symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-neurify_symbolic.log

#==================#
# deeppoly+neurify #
#==================#

$1 tests/census/census.txt tests/census/20.py --domain deeppoly_neurify --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-deeppoly_neurify.log

#========================#
# boxes+deeppoly+neurify #
#========================#

$1 tests/census/census.txt tests/census/20.py --domain boxes_deeppoly_neurify --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-boxes_deeppoly_neurify.log

#===========================#
# deeppoly+neurify+symbolic #
#===========================#

$1 tests/census/census.txt tests/census/20.py --domain deeppoly_neurify_symbolic --min_lower 0 --lower 1 --upper 0 --max_upper 20 | tee tests/census/logs4/census-20-deeppoly_neurify_symbolic.log
