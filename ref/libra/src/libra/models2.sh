#!/usr/bin/env bash
############################################
# experiment B: different model structures #
############################################

LIBRA=$@

#=======#
# L=0.5 #
#=======#

#----#
# 10 #
#----#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes_deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-neurify_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes_deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly_neurify_symbolic-0.5-5.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes_deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-neurify_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-boxes_deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-10-deeppoly_neurify_symbolic-0.5-5.log
fi

#---------#
# 12
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes_deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-neurify_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes_deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly_neurify_symbolic-0.5-5.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes_deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-neurify_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-boxes_deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-12-deeppoly_neurify_symbolic-0.5-5.log
fi

#---------#
# 20
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes_deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-neurify_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes_deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly_neurify_symbolic-0.5-5.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes_deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-neurify_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-boxes_deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-20-deeppoly_neurify_symbolic-0.5-5.log
fi

#---------#
# 40
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes_deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-neurify_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes_deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly_neurify_symbolic-0.5-5.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes_deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-neurify_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-boxes_deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-40-deeppoly_neurify_symbolic-0.5-5.log
fi

#---------#
# 45
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes_deeppoly-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-neurify_symbolic-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes_deeppoly_neurify-0.5-5.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly_neurify_symbolic-0.5-5.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes_deeppoly --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes_deeppoly-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-neurify_symbolic-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes_deeppoly_neurify --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-boxes_deeppoly_neurify-0.5-5.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly_neurify_symbolic --lower 0.5 --upper 5 | tee tests/census/logs2/census-45-deeppoly_neurify_symbolic-0.5-5.log
fi
