#!/usr/bin/env bash
#################################
# experiment #1: detecting bias #
#################################

LIBRA=$@

#=======#
# boxes #
#=======#

#----------------------#
# assume (x05 <= 0.04) #
#----------------------#

$@ tests/german/german.txt tests/german/LtEno-bias1.py --domain boxes --lower 0 --upper 8 | tee tests/german/logs/german-LtEno-bias1-boxes-0-8.log
$@ tests/german/german.txt tests/german/LtEno-bias2.py --domain boxes --lower 0 --upper 9 | tee tests/german/logs/german-LtEno-bias2-boxes-0-9.log
$@ tests/german/german.txt tests/german/LtEno-bias3.py --domain boxes --lower 0 --upper 2 | tee tests/german/logs/german-LtEno-bias3-boxes-0-2.log
$@ tests/german/german.txt tests/german/LtEno-bias4.py --domain boxes --lower 0 --upper 13 | tee tests/german/logs/german-LtEno-bias4-boxes-0-13.log
$@ tests/german/german.txt tests/german/LtEno-bias5.py --domain boxes --lower 0 --upper 6 | tee tests/german/logs/german-LtEno-bias5-boxes-0-6.log
$@ tests/german/german.txt tests/german/LtEno-bias6.py --domain boxes --lower 0 --upper 13 | tee tests/german/logs/german-LtEno-bias6-boxes-0-13.log
$@ tests/german/german.txt tests/german/LtEno-bias7.py --domain boxes --lower 0 --upper 9 | tee tests/german/logs/german-LtEno-bias7-boxes-0-9.log
$@ tests/german/german.txt tests/german/LtEno-bias8.py --domain boxes --lower 0 --upper 6 | tee tests/german/logs/german-LtEno-bias8-boxes-0-6.log

$@ tests/german/german.txt tests/german/LtE0.20-bias1.py --domain boxes --lower 0 --upper 13 | tee tests/german/logs/german-LtE0.20-bias1-boxes-0-13.log
$@ tests/german/german.txt tests/german/LtE0.20-bias2.py --domain boxes --lower 0 --upper 5 | tee tests/german/logs/german-LtE0.20-bias2-boxes-0-5.log
$@ tests/german/german.txt tests/german/LtE0.20-bias3.py --domain boxes --lower 0 --upper 12 | tee tests/german/logs/german-LtE0.20-bias3-boxes-0-12.log
$@ tests/german/german.txt tests/german/LtE0.20-bias4.py --domain boxes --lower 0 --upper 9 | tee tests/german/logs/german-LtE0.20-bias4-boxes-0-9.log
$@ tests/german/german.txt tests/german/LtE0.20-bias5.py --domain boxes --lower 0 --upper 10 | tee tests/german/logs/german-LtE0.20-bias5-boxes-0-10.log
$@ tests/german/german.txt tests/german/LtE0.20-bias6.py --domain boxes --lower 0 --upper 7 | tee tests/german/logs/german-LtE0.20-bias6-boxes-0-7.log
$@ tests/german/german.txt tests/german/LtE0.20-bias7.py --domain boxes --lower 0 --upper 9 | tee tests/german/logs/german-LtE0.20-bias7-boxes-0-9.log
$@ tests/german/german.txt tests/german/LtE0.20-bias8.py --domain boxes --lower 0 --upper 3 | tee tests/german/logs/german-LtE0.20-bias8-boxes-0-3.log

##=====================#
## assume (x05 > 0.04) #
##=====================#

$@ tests/german/german.txt tests/german/Gtno-bias1.py --domain boxes --lower 0 --upper 13 | tee tests/german/logs/german-Gtno-bias1-boxes-0-13.log
$@ tests/german/german.txt tests/german/Gtno-bias2.py --domain boxes --lower 0 --upper 15 | tee tests/german/logs/german-Gtno-bias2-boxes-0-15.log
$@ tests/german/german.txt tests/german/Gtno-bias3.py --domain boxes --lower 0 --upper 3 | tee tests/german/logs/german-Gtno-bias3-boxes-0-3.log
$@ tests/german/german.txt tests/german/Gtno-bias4.py --domain boxes --lower 0 --upper 13 | tee tests/german/logs/german-Gtno-bias4-boxes-0-13.log
$@ tests/german/german.txt tests/german/Gtno-bias5.py --domain boxes --lower 0 --upper 7 | tee tests/german/logs/german-Gtno-bias5-boxes-0-7.log
$@ tests/german/german.txt tests/german/Gtno-bias6.py --domain boxes --lower 0 --upper 16 | tee tests/german/logs/german-Gtno-bias6-boxes-0-16.log
$@ tests/german/german.txt tests/german/Gtno-bias7.py --domain boxes --lower 0 --upper 10 | tee tests/german/logs/german-Gtno-bias7-boxes-0-10.log
$@ tests/german/german.txt tests/german/Gtno-bias8.py --domain boxes --lower 0 --upper 10 | tee tests/german/logs/german-Gtno-bias8-boxes-0-10.log

$@ tests/german/german.txt tests/german/Gt0.20-bias1.py --domain boxes --lower 0 --upper 16 | tee tests/german/logs/german-Gt0.20-bias1-boxes-0-16.log
$@ tests/german/german.txt tests/german/Gt0.20-bias2.py --domain boxes --lower 0 --upper 10 | tee tests/german/logs/german-Gt0.20-bias2-boxes-0-10.log
$@ tests/german/german.txt tests/german/Gt0.20-bias3.py --domain boxes --lower 0 --upper 16 | tee tests/german/logs/german-Gt0.20-bias3-boxes-0-16.log
$@ tests/german/german.txt tests/german/Gt0.20-bias4.py --domain boxes --lower 0 --upper 10 | tee tests/german/logs/german-Gt0.20-bias4-boxes-0-10.log
$@ tests/german/german.txt tests/german/Gt0.20-bias5.py --domain boxes --lower 0 --upper 14 | tee tests/german/logs/german-Gt0.20-bias5-boxes-0-14.log
$@ tests/german/german.txt tests/german/Gt0.20-bias6.py --domain boxes --lower 0 --upper 9 | tee tests/german/logs/german-Gt0.20-bias6-boxes-0-9.log
$@ tests/german/german.txt tests/german/Gt0.20-bias7.py --domain boxes --lower 0 --upper 11 | tee tests/german/logs/german-Gt0.20-bias7-boxes-0-11.log
$@ tests/german/german.txt tests/german/Gt0.20-bias8.py --domain boxes --lower 0 --upper 3 | tee tests/german/logs/german-Gt0.20-bias8-boxes-0-3.log

#==========#
# symbolic #
#==========#

#----------------------#
# assume (x05 <= 0.04) #
#----------------------#

$@ tests/german/german.txt tests/german/LtEno-bias1.py --domain symbolic --lower 0 --upper 7 | tee tests/german/logs/german-LtEno-bias1-symbolic-0-7.log
$@ tests/german/german.txt tests/german/LtEno-bias2.py --domain symbolic --lower 0 --upper 6 | tee tests/german/logs/german-LtEno-bias2-symbolic-0-6.log
$@ tests/german/german.txt tests/german/LtEno-bias3.py --domain symbolic --lower 0 --upper 2 | tee tests/german/logs/german-LtEno-bias3-symbolic-0-2.log
$@ tests/german/german.txt tests/german/LtEno-bias4.py --domain symbolic --lower 0 --upper 9 | tee tests/german/logs/german-LtEno-bias4-symbolic-0-9.log
$@ tests/german/german.txt tests/german/LtEno-bias5.py --domain symbolic --lower 0 --upper 3 | tee tests/german/logs/german-LtEno-bias5-symbolic-0-3.log
$@ tests/german/german.txt tests/german/LtEno-bias6.py --domain symbolic --lower 0 --upper 8 | tee tests/german/logs/german-LtEno-bias6-symbolic-0-8.log
$@ tests/german/german.txt tests/german/LtEno-bias7.py --domain symbolic --lower 0 --upper 6 | tee tests/german/logs/german-LtEno-bias7-symbolic-0-6.log
$@ tests/german/german.txt tests/german/LtEno-bias8.py --domain symbolic --lower 0 --upper 5 | tee tests/german/logs/german-LtEno-bias8-symbolic-0-5.log

$@ tests/german/german.txt tests/german/LtE0.20-bias1.py --domain symbolic --lower 0 --upper 10 | tee tests/german/logs/german-LtE0.20-bias1-symbolic-0-10.log
$@ tests/german/german.txt tests/german/LtE0.20-bias2.py --domain symbolic --lower 0 --upper 4 | tee tests/german/logs/german-LtE0.20-bias2-symbolic-0-4.log
$@ tests/german/german.txt tests/german/LtE0.20-bias3.py --domain symbolic --lower 0 --upper 12 | tee tests/german/logs/german-LtE0.20-bias3-symbolic-0-12.log
$@ tests/german/german.txt tests/german/LtE0.20-bias4.py --domain symbolic --lower 0 --upper 5 | tee tests/german/logs/german-LtE0.20-bias4-symbolic-0-5.log
$@ tests/german/german.txt tests/german/LtE0.20-bias5.py --domain symbolic --lower 0 --upper 8 | tee tests/german/logs/german-LtE0.20-bias5-symbolic-0-8.log
$@ tests/german/german.txt tests/german/LtE0.20-bias6.py --domain symbolic --lower 0 --upper 2 | tee tests/german/logs/german-LtE0.20-bias6-symbolic-0-2.log
$@ tests/german/german.txt tests/german/LtE0.20-bias7.py --domain symbolic --lower 0 --upper 12 | tee tests/german/logs/german-LtE0.20-bias7-symbolic-0-12.log
$@ tests/german/german.txt tests/german/LtE0.20-bias8.py --domain symbolic --lower 0 --upper 2 | tee tests/german/logs/german-LtE0.20-bias8-symbolic-0-2.log

##=====================#
## assume (x05 > 0.04) #
##=====================#

$@ tests/german/german.txt tests/german/Gtno-bias1.py --domain symbolic --lower 0 --upper 12 | tee tests/german/logs/german-Gtno-bias1-symbolic-0-12.log
$@ tests/german/german.txt tests/german/Gtno-bias2.py --domain symbolic --lower 0 --upper 15 | tee tests/german/logs/german-Gtno-bias2-symbolic-0-15.log
$@ tests/german/german.txt tests/german/Gtno-bias3.py --domain symbolic --lower 0 --upper 2 | tee tests/german/logs/german-Gtno-bias3-symbolic-0-2.log
$@ tests/german/german.txt tests/german/Gtno-bias4.py --domain symbolic --lower 0 --upper 9 | tee tests/german/logs/german-Gtno-bias4-symbolic-0-9.log
$@ tests/german/german.txt tests/german/Gtno-bias5.py --domain symbolic --lower 0 --upper 3 | tee tests/german/logs/german-Gtno-bias5-symbolic-0-3.log
$@ tests/german/german.txt tests/german/Gtno-bias6.py --domain symbolic --lower 0 --upper 16 | tee tests/german/logs/german-Gtno-bias6-symbolic-0-16.log
$@ tests/german/german.txt tests/german/Gtno-bias7.py --domain symbolic --lower 0 --upper 8 | tee tests/german/logs/german-Gtno-bias7-symbolic-0-8.log
$@ tests/german/german.txt tests/german/Gtno-bias8.py --domain symbolic --lower 0 --upper 9 | tee tests/german/logs/german-Gtno-bias8-symbolic-0-9.log

$@ tests/german/german.txt tests/german/Gt0.20-bias1.py --domain symbolic --lower 0 --upper 13 | tee tests/german/logs/german-Gt0.20-bias1-symbolic-0-13.log
$@ tests/german/german.txt tests/german/Gt0.20-bias2.py --domain symbolic --lower 0 --upper 6 | tee tests/german/logs/german-Gt0.20-bias2-symbolic-0-6.log
$@ tests/german/german.txt tests/german/Gt0.20-bias3.py --domain symbolic --lower 0 --upper 15 | tee tests/german/logs/german-Gt0.20-bias3-symbolic-0-15.log
$@ tests/german/german.txt tests/german/Gt0.20-bias4.py --domain symbolic --lower 0 --upper 8 | tee tests/german/logs/german-Gt0.20-bias4-symbolic-0-8.log
$@ tests/german/german.txt tests/german/Gt0.20-bias5.py --domain symbolic --lower 0 --upper 14 | tee tests/german/logs/german-Gt0.20-bias5-symbolic-0-14.log
$@ tests/german/german.txt tests/german/Gt0.20-bias6.py --domain symbolic --lower 0 --upper 6 | tee tests/german/logs/german-Gt0.20-bias6-symbolic-0-6.log
$@ tests/german/german.txt tests/german/Gt0.20-bias7.py --domain symbolic --lower 0 --upper 6 | tee tests/german/logs/german-Gt0.20-bias7-symbolic-0-6.log
$@ tests/german/german.txt tests/german/Gt0.20-bias8.py --domain symbolic --lower 0 --upper 2 | tee tests/german/logs/german-Gt0.20-bias8-symbolic-0-2.log

#=======#
# deeppoly #
#=======#

#----------------------#
# assume (x05 <= 0.04) #
#----------------------#

$@ tests/german/german.txt tests/german/LtEno-bias1.py --domain deeppoly --lower 0 --upper 8 | tee tests/german/logs/german-LtEno-bias1-deeppoly-0-8.log
$@ tests/german/german.txt tests/german/LtEno-bias2.py --domain deeppoly --lower 0 --upper 6 | tee tests/german/logs/german-LtEno-bias2-deeppoly-0-6.log
$@ tests/german/german.txt tests/german/LtEno-bias3.py --domain deeppoly --lower 0 --upper 2 | tee tests/german/logs/german-LtEno-bias3-deeppoly-0-2.log
$@ tests/german/german.txt tests/german/LtEno-bias4.py --domain deeppoly --lower 0 --upper 7 | tee tests/german/logs/german-LtEno-bias4-deeppoly-0-7.log
$@ tests/german/german.txt tests/german/LtEno-bias5.py --domain deeppoly --lower 0 --upper 3 | tee tests/german/logs/german-LtEno-bias5-deeppoly-0-3.log
$@ tests/german/german.txt tests/german/LtEno-bias6.py --domain deeppoly --lower 0 --upper 12 | tee tests/german/logs/german-LtEno-bias6-deeppoly-0-12.log
$@ tests/german/german.txt tests/german/LtEno-bias7.py --domain deeppoly --lower 0 --upper 6 | tee tests/german/logs/german-LtEno-bias7-deeppoly-0-6.log
$@ tests/german/german.txt tests/german/LtEno-bias8.py --domain deeppoly --lower 0 --upper 5 | tee tests/german/logs/german-LtEno-bias8-deeppoly-0-5.log

$@ tests/german/german.txt tests/german/LtE0.20-bias1.py --domain deeppoly --lower 0 --upper 8 | tee tests/german/logs/german-LtE0.20-bias1-deeppoly-0-8.log
$@ tests/german/german.txt tests/german/LtE0.20-bias2.py --domain deeppoly --lower 0 --upper 4 | tee tests/german/logs/german-LtE0.20-bias2-deeppoly-0-4.log
$@ tests/german/german.txt tests/german/LtE0.20-bias3.py --domain deeppoly --lower 0 --upper 12 | tee tests/german/logs/german-LtE0.20-bias3-deeppoly-0-12.log
$@ tests/german/german.txt tests/german/LtE0.20-bias4.py --domain deeppoly --lower 0 --upper 4 | tee tests/german/logs/german-LtE0.20-bias4-deeppoly-0-4.log
$@ tests/german/german.txt tests/german/LtE0.20-bias5.py --domain deeppoly --lower 0 --upper 10 | tee tests/german/logs/german-LtE0.20-bias5-deeppoly-0-10.log
$@ tests/german/german.txt tests/german/LtE0.20-bias6.py --domain deeppoly --lower 0 --upper 2 | tee tests/german/logs/german-LtE0.20-bias6-deeppoly-0-2.log
$@ tests/german/german.txt tests/german/LtE0.20-bias7.py --domain deeppoly --lower 0 --upper 3 | tee tests/german/logs/german-LtE0.20-bias7-deeppoly-0-3.log
$@ tests/german/german.txt tests/german/LtE0.20-bias8.py --domain deeppoly --lower 0 --upper 1 | tee tests/german/logs/german-LtE0.20-bias8-deeppoly-0-1.log

##=====================#
## assume (x05 > 0.04) #
##=====================#

$@ tests/german/german.txt tests/german/Gtno-bias1.py --domain deeppoly --lower 0 --upper 10 | tee tests/german/logs/german-Gtno-bias1-deeppoly-0-10.log
$@ tests/german/german.txt tests/german/Gtno-bias2.py --domain deeppoly --lower 0 --upper 11 | tee tests/german/logs/german-Gtno-bias2-deeppoly-0-11.log
$@ tests/german/german.txt tests/german/Gtno-bias3.py --domain deeppoly --lower 0 --upper 2 | tee tests/german/logs/german-Gtno-bias3-deeppoly-0-2.log
$@ tests/german/german.txt tests/german/Gtno-bias4.py --domain deeppoly --lower 0 --upper 10 | tee tests/german/logs/german-Gtno-bias4-deeppoly-0-10.log
$@ tests/german/german.txt tests/german/Gtno-bias5.py --domain deeppoly --lower 0 --upper 4 | tee tests/german/logs/german-Gtno-bias5-deeppoly-0-4.log
$@ tests/german/german.txt tests/german/Gtno-bias6.py --domain deeppoly --lower 0 --upper 14 | tee tests/german/logs/german-Gtno-bias6-deeppoly-0-14.log
$@ tests/german/german.txt tests/german/Gtno-bias7.py --domain deeppoly --lower 0 --upper 7 | tee tests/german/logs/german-Gtno-bias7-deeppoly-0-7.log
$@ tests/german/german.txt tests/german/Gtno-bias8.py --domain deeppoly --lower 0 --upper 9 | tee tests/german/logs/german-Gtno-bias8-deeppoly-0-9.log

$@ tests/german/german.txt tests/german/Gt0.20-bias1.py --domain deeppoly --lower 0 --upper 16 | tee tests/german/logs/german-Gt0.20-bias1-deeppoly-0-11.log
$@ tests/german/german.txt tests/german/Gt0.20-bias2.py --domain deeppoly --lower 0 --upper 10 | tee tests/german/logs/german-Gt0.20-bias2-deeppoly-0-7.log
$@ tests/german/german.txt tests/german/Gt0.20-bias3.py --domain deeppoly --lower 0 --upper 16 | tee tests/german/logs/german-Gt0.20-bias3-deeppoly-0-7.log
$@ tests/german/german.txt tests/german/Gt0.20-bias4.py --domain deeppoly --lower 0 --upper 10 | tee tests/german/logs/german-Gt0.20-bias4-deeppoly-0-6.log
$@ tests/german/german.txt tests/german/Gt0.20-bias5.py --domain deeppoly --lower 0 --upper 14 | tee tests/german/logs/german-Gt0.20-bias5-deeppoly-0-13.log
$@ tests/german/german.txt tests/german/Gt0.20-bias6.py --domain deeppoly --lower 0 --upper 9 | tee tests/german/logs/german-Gt0.20-bias6-deeppoly-0-5.log
$@ tests/german/german.txt tests/german/Gt0.20-bias7.py --domain deeppoly --lower 0 --upper 11 | tee tests/german/logs/german-Gt0.20-bias7-deeppoly-0-8.log
$@ tests/german/german.txt tests/german/Gt0.20-bias8.py --domain deeppoly --lower 0 --upper 3 | tee tests/german/logs/german-Gt0.20-bias8-deeppoly-0-2.log
