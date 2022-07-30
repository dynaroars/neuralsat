#!/usr/bin/env bash
##############################
# experiment 2: bias queries #
##############################

LIBRA=$@

#=======#
# boxes #
#=======#

#-----------------------#
# age < 25              #
# assume(0 <= x02 <= 0) #
# assume(1 <= x03 <= 1) #
# assume(0 <= x04 <= 0) #
# race bias             #
#-----------------------#

$@ tests/compas/race.txt tests/compas/Lt25no-bias1.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias1-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias2.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias2-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias3.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias3-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias4.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias4-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias5.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias5-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias6.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias6-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias7.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias7-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias8.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias8-boxes-0-10.log

$@ tests/compas/race.txt tests/compas/Lt250.20-bias1.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias1-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias2.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias2-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias3.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias3-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias4.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias4-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias5.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias5-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias6.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias6-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias7.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias7-boxes-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias8.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias8-boxes-0-10.log

#-----------------------#
# male                  #
# assume(0 <= x00 <= 0) #
# assume(1 <= x01 <= 1) #
# age bias              #
#-----------------------#

$@ tests/compas/age.txt tests/compas/Mno-bias1.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias1-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias2.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias2-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias3.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias3-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias4.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias4-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias5.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias5-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias6.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias6-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias7.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias7-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias8.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias8-boxes-0-10.log

$@ tests/compas/age.txt tests/compas/M0.20-bias1.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias1-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias2.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias2-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias3.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias3-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias4.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias4-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias5.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias5-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias6.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias6-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias7.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias7-boxes-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias8.py --domain boxes --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias8-boxes-0-10.log

#------------------------#
# caucasian              #
# assume(0 <= x05 <= 0)  #
# assume(0 <= x06 <= 0)  #
# assume(0 <= x07 <= 0)  #
# assume(0 <= x08 <= 0)  #
# assume(0 <= x09 <= 0)  #
# assume(1 <= x010 <= 1) #
# priors bias            #
#------------------------#

$@ tests/compas/priors.txt tests/compas/Cno-bias1.py --domain boxes --lower 0 --upper 12 | tee tests/compas/logs/compas-Cno-bias1-boxes-0-12.log
$@ tests/compas/priors.txt tests/compas/Cno-bias2.py --domain boxes --lower 0 --upper 14 | tee tests/compas/logs/compas-Cno-bias2-boxes-0-14.log
$@ tests/compas/priors.txt tests/compas/Cno-bias3.py --domain boxes --lower 0 --upper 14 | tee tests/compas/logs/compas-Cno-bias3-boxes-0-14.log
$@ tests/compas/priors.txt tests/compas/Cno-bias4.py --domain boxes --lower 0 --upper 18 | tee tests/compas/logs/compas-Cno-bias4-boxes-0-18.log
$@ tests/compas/priors.txt tests/compas/Cno-bias5.py --domain boxes --lower 0 --upper 19 | tee tests/compas/logs/compas-Cno-bias5-boxes-0-19.log
$@ tests/compas/priors.txt tests/compas/Cno-bias6.py --domain boxes --lower 0 --upper 12 | tee tests/compas/logs/compas-Cno-bias6-boxes-0-12.log
$@ tests/compas/priors.txt tests/compas/Cno-bias7.py --domain boxes --lower 0 --upper 13 | tee tests/compas/logs/compas-Cno-bias7-boxes-0-13.log
$@ tests/compas/priors.txt tests/compas/Cno-bias8.py --domain boxes --lower 0 --upper 15 | tee tests/compas/logs/compas-Cno-bias8-boxes-0-15.log

$@ tests/compas/priors.txt tests/compas/C0.20-bias1.py --domain boxes --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias1-boxes-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias2.py --domain boxes --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias2-boxes-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias3.py --domain boxes --lower 0 --upper 13 | tee tests/compas/logs/compas-C0.20-bias3-boxes-0-13.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias4.py --domain boxes --lower 0 --upper 9 | tee tests/compas/logs/compas-C0.20-bias4-boxes-0-9.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias5.py --domain boxes --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias5-boxes-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias6.py --domain boxes --lower 0 --upper 15 | tee tests/compas/logs/compas-C0.20-bias6-boxes-0-15.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias7.py --domain boxes --lower 0 --upper 19 | tee tests/compas/logs/compas-C0.20-bias7-boxes-0-19.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias8.py --domain boxes --lower 0 --upper 17 | tee tests/compas/logs/compas-C0.20-bias8-boxes-0-17.log

#==========#
# symbolic #
#==========#

#-----------------------#
# age < 25              #
# assume(0 <= x02 <= 0) #
# assume(1 <= x03 <= 1) #
# assume(0 <= x04 <= 0) #
# race bias             #
#-----------------------#

$@ tests/compas/race.txt tests/compas/Lt25no-bias1.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias1-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias2.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias2-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias3.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias3-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias4.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias4-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias5.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias5-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias6.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias6-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias7.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias7-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias8.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias8-symbolic-0-10.log

$@ tests/compas/race.txt tests/compas/Lt250.20-bias1.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias1-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias2.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias2-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias3.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias3-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias4.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias4-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias5.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias5-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias6.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias6-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias7.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias7-symbolic-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias8.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias8-symbolic-0-10.log

#-----------------------#
# male                  #
# assume(0 <= x00 <= 0) #
# assume(1 <= x01 <= 1) #
# age bias              #
#-----------------------#

$@ tests/compas/age.txt tests/compas/Mno-bias1.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias1-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias2.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias2-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias3.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias3-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias4.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias4-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias5.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias5-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias6.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias6-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias7.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias7-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias8.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias8-symbolic-0-10.log

$@ tests/compas/age.txt tests/compas/M0.20-bias1.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias1-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias2.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias2-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias3.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias3-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias4.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias4-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias5.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias5-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias6.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias6-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias7.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias7-symbolic-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias8.py --domain symbolic --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias8-symbolic-0-10.log

#------------------------#
# caucasian              #
# assume(0 <= x05 <= 0)  #
# assume(0 <= x06 <= 0)  #
# assume(0 <= x07 <= 0)  #
# assume(0 <= x08 <= 0)  #
# assume(0 <= x09 <= 0)  #
# assume(1 <= x010 <= 1) #
# priors bias            #
#------------------------#

$@ tests/compas/priors.txt tests/compas/Cno-bias1.py --domain symbolic --lower 0 --upper 12 | tee tests/compas/logs/compas-Cno-bias1-symbolic-0-12.log
$@ tests/compas/priors.txt tests/compas/Cno-bias2.py --domain symbolic --lower 0 --upper 12 | tee tests/compas/logs/compas-Cno-bias2-symbolic-0-12.log
$@ tests/compas/priors.txt tests/compas/Cno-bias3.py --domain symbolic --lower 0 --upper 14 | tee tests/compas/logs/compas-Cno-bias3-symbolic-0-14.log
$@ tests/compas/priors.txt tests/compas/Cno-bias4.py --domain symbolic --lower 0 --upper 18 | tee tests/compas/logs/compas-Cno-bias4-symbolic-0-18.log
$@ tests/compas/priors.txt tests/compas/Cno-bias5.py --domain symbolic --lower 0 --upper 19 | tee tests/compas/logs/compas-Cno-bias5-symbolic-0-19.log
$@ tests/compas/priors.txt tests/compas/Cno-bias6.py --domain symbolic --lower 0 --upper 12 | tee tests/compas/logs/compas-Cno-bias6-symbolic-0-12.log
$@ tests/compas/priors.txt tests/compas/Cno-bias7.py --domain symbolic --lower 0 --upper 13 | tee tests/compas/logs/compas-Cno-bias7-symbolic-0-13.log
$@ tests/compas/priors.txt tests/compas/Cno-bias8.py --domain symbolic --lower 0 --upper 15 | tee tests/compas/logs/compas-Cno-bias8-symbolic-0-15.log

$@ tests/compas/priors.txt tests/compas/C0.20-bias1.py --domain symbolic --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias1-symbolic-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias2.py --domain symbolic --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias2-symbolic-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias3.py --domain symbolic --lower 0 --upper 12 | tee tests/compas/logs/compas-C0.20-bias3-symbolic-0-12.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias4.py --domain symbolic --lower 0 --upper 9 | tee tests/compas/logs/compas-C0.20-bias4-symbolic-0-9.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias5.py --domain symbolic --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias5-symbolic-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias6.py --domain symbolic --lower 0 --upper 15 | tee tests/compas/logs/compas-C0.20-bias6-symbolic-0-15.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias7.py --domain symbolic --lower 0 --upper 19 | tee tests/compas/logs/compas-C0.20-bias7-symbolic-0-19.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias8.py --domain symbolic --lower 0 --upper 17 | tee tests/compas/logs/compas-C0.20-bias8-symbolic-0-17.log

#==========#
# deeppoly #
#==========#

#-----------------------#
# age < 25              #
# assume(0 <= x02 <= 0) #
# assume(1 <= x03 <= 1) #
# assume(0 <= x04 <= 0) #
# race bias             #
#-----------------------#

$@ tests/compas/race.txt tests/compas/Lt25no-bias1.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias1-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias2.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias2-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias3.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias3-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias4.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias4-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias5.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias5-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias6.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias6-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias7.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias7-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt25no-bias8.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt25no-bias8-deeppoly-0-10.log

$@ tests/compas/race.txt tests/compas/Lt250.20-bias1.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias1-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias2.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias2-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias3.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias3-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias4.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias4-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias5.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias5-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias6.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias6-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias7.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias7-deeppoly-0-10.log
$@ tests/compas/race.txt tests/compas/Lt250.20-bias8.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Lt250.20-bias8-deeppoly-0-10.log

#-----------------------#
# male                  #
# assume(0 <= x00 <= 0) #
# assume(1 <= x01 <= 1) #
# age bias              #
#-----------------------#

$@ tests/compas/age.txt tests/compas/Mno-bias1.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias1-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias2.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias2-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias3.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias3-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias4.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias4-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias5.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias5-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias6.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias6-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias7.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias7-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/Mno-bias8.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-Mno-bias8-deeppoly-0-10.log

$@ tests/compas/age.txt tests/compas/M0.20-bias1.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias1-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias2.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias2-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias3.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias3-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias4.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias4-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias5.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias5-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias6.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias6-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias7.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias7-deeppoly-0-10.log
$@ tests/compas/age.txt tests/compas/M0.20-bias8.py --domain deeppoly --lower 0 --upper 10 | tee tests/compas/logs/compas-M0.20-bias8-deeppoly-0-10.log

#------------------------#
# caucasian              #
# assume(0 <= x05 <= 0)  #
# assume(0 <= x06 <= 0)  #
# assume(0 <= x07 <= 0)  #
# assume(0 <= x08 <= 0)  #
# assume(0 <= x09 <= 0)  #
# assume(1 <= x010 <= 1) #
# priors bias            #
#------------------------#

$@ tests/compas/priors.txt tests/compas/Cno-bias1.py --domain deeppoly --lower 0 --upper 11 | tee tests/compas/logs/compas-Cno-bias1-deeppoly-0-11.log
$@ tests/compas/priors.txt tests/compas/Cno-bias2.py --domain deeppoly --lower 0 --upper 7 | tee tests/compas/logs/compas-Cno-bias2-deeppoly-0-7.log
$@ tests/compas/priors.txt tests/compas/Cno-bias3.py --domain deeppoly --lower 0 --upper 11 | tee tests/compas/logs/compas-Cno-bias3-deeppoly-0-11.log
$@ tests/compas/priors.txt tests/compas/Cno-bias4.py --domain deeppoly --lower 0 --upper 17 | tee tests/compas/logs/compas-Cno-bias4-deeppoly-0-17.log
$@ tests/compas/priors.txt tests/compas/Cno-bias5.py --domain deeppoly --lower 0 --upper 19 | tee tests/compas/logs/compas-Cno-bias5-deeppoly-0-19.log
$@ tests/compas/priors.txt tests/compas/Cno-bias6.py --domain deeppoly --lower 0 --upper 11 | tee tests/compas/logs/compas-Cno-bias6-deeppoly-0-11.log
$@ tests/compas/priors.txt tests/compas/Cno-bias7.py --domain deeppoly --lower 0 --upper 15 | tee tests/compas/logs/compas-Cno-bias7-deeppoly-0-15.log
$@ tests/compas/priors.txt tests/compas/Cno-bias8.py --domain deeppoly --lower 0 --upper 15 | tee tests/compas/logs/compas-Cno-bias8-deeppoly-0-15.log

$@ tests/compas/priors.txt tests/compas/C0.20-bias1.py --domain deeppoly --lower 0 --upper 11 | tee tests/compas/logs/compas-C0.20-bias1-deeppoly-0-11.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias2.py --domain deeppoly --lower 0 --upper 11 | tee tests/compas/logs/compas-C0.20-bias2-deeppoly-0-11.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias3.py --domain deeppoly --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias3-deeppoly-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias4.py --domain deeppoly --lower 0 --upper 7 | tee tests/compas/logs/compas-C0.20-bias4-deeppoly-0-7.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias5.py --domain deeppoly --lower 0 --upper 13 | tee tests/compas/logs/compas-C0.20-bias5-deeppoly-0-13.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias6.py --domain deeppoly --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias6-deeppoly-0-14.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias7.py --domain deeppoly --lower 0 --upper 17 | tee tests/compas/logs/compas-C0.20-bias7-deeppoly-0-17.log
$@ tests/compas/priors.txt tests/compas/C0.20-bias8.py --domain deeppoly --lower 0 --upper 14 | tee tests/compas/logs/compas-C0.20-bias8-deeppoly-0-14.log
