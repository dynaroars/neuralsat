#!/usr/bin/env bash
############################################
# experiment 3: different model structures #
############################################

LIBRA=$@

#=======#
# L=0.5 #
#=======#

#----#
# 10 #
#----#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-10-boxes-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-10-boxes-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-10-boxes-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-10-boxes-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-10-symbolic-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-10-symbolic-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-10-symbolic-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-10-symbolic-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-10-deeppoly-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-10-deeppoly-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-10-deeppoly-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-10-deeppoly-0.5-10.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-10-boxes-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-10-boxes-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-10-boxes-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-10-boxes-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-10-symbolic-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-10-symbolic-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-10-symbolic-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-10-symbolic-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-10-deeppoly-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-10-deeppoly-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-10-deeppoly-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/10.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-10-deeppoly-0.5-10.log
fi

#---------#
# 12
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-12-boxes-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-12-boxes-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-12-boxes-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-12-boxes-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-12-symbolic-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-12-symbolic-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-12-symbolic-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-12-symbolic-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-12-deeppoly-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-12-deeppoly-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-12-deeppoly-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-12-deeppoly-0.5-10.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-12-boxes-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-12-boxes-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-12-boxes-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-12-boxes-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-12-symbolic-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-12-symbolic-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-12-symbolic-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-12-symbolic-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-12-deeppoly-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-12-deeppoly-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-12-deeppoly-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/12.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-12-deeppoly-0.5-10.log
fi


#---------#
# 20
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-20-boxes-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-20-boxes-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-20-boxes-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-20-boxes-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-20-symbolic-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-20-symbolic-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-20-symbolic-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-20-symbolic-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-20-deeppoly-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-20-deeppoly-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-20-deeppoly-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-20-deeppoly-0.5-10.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-20-boxes-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-20-boxes-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-20-boxes-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-20-boxes-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-20-symbolic-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-20-symbolic-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-20-symbolic-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-20-symbolic-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-20-deeppoly-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-20-deeppoly-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-20-deeppoly-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/20.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-20-deeppoly-0.5-10.log
fi

#---------#
# 40
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-40-boxes-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-40-boxes-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-40-boxes-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-40-boxes-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-40-symbolic-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-40-symbolic-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-40-symbolic-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-40-symbolic-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-40-deeppoly-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-40-deeppoly-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-40-deeppoly-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-40-deeppoly-0.5-10.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-40-boxes-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-40-boxes-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-40-boxes-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-40-boxes-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-40-symbolic-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-40-symbolic-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-40-symbolic-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-40-symbolic-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-40-deeppoly-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-40-deeppoly-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-40-deeppoly-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/40.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-40-deeppoly-0.5-10.log
fi

#---------#
# 45
#---------#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-45-boxes-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-45-boxes-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-45-boxes-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-45-boxes-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-45-symbolic-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-45-symbolic-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-45-symbolic-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-45-symbolic-0.5-10.log
    
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-45-deeppoly-0.5-4.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-45-deeppoly-0.5-6.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-45-deeppoly-0.5-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-45-deeppoly-0.5-10.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 4 | tee tests/census/logs1/census-45-boxes-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 6 | tee tests/census/logs1/census-45-boxes-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 8 | tee tests/census/logs1/census-45-boxes-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain boxes --lower 0.5 --upper 10 | tee tests/census/logs1/census-45-boxes-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 4 | tee tests/census/logs1/census-45-symbolic-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 6 | tee tests/census/logs1/census-45-symbolic-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 8 | tee tests/census/logs1/census-45-symbolic-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain symbolic --lower 0.5 --upper 10 | tee tests/census/logs1/census-45-symbolic-0.5-10.log
    
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 4 | tee tests/census/logs1/census-45-deeppoly-0.5-4.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 6 | tee tests/census/logs1/census-45-deeppoly-0.5-6.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 8 | tee tests/census/logs1/census-45-deeppoly-0.5-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/45.py --domain deeppoly --lower 0.5 --upper 10 | tee tests/census/logs1/census-45-deeppoly-0.5-10.log
fi
