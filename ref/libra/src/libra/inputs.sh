#!/usr/bin/env bash
#######################################
# experiment 3: different input space #
#######################################

LIBRA=$@

#=======#
# boxes #
#=======#

#---#
# F #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20F.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20F-boxes-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80F.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80F-boxes-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320F.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320F-boxes-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280F.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280F-boxes-0.25-128.log    
else
    timeout 46800 $@ tests/census/census.txt tests/census/20F.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20F-boxes-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80F.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80F-boxes-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320F.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320F-boxes-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280F.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280F-boxes-0.25-128.log    
fi

#---#
# E #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20E.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20E-boxes-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80E.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80E-boxes-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320E.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320E-boxes-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280E.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280E-boxes-0.25-128.log    
else
    timeout 46800 $@ tests/census/census.txt tests/census/20E.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20E-boxes-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80E.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80E-boxes-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320E.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320E-boxes-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280E.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280E-boxes-0.25-128.log    
fi

#---#
# D #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20D.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20D-boxes-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80D.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80D-boxes-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320D.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320D-boxes-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280D.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280D-boxes-0.25-128.log    
else
    timeout 46800 $@ tests/census/census.txt tests/census/20D.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20D-boxes-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80D.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80D-boxes-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320D.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320D-boxes-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280D.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280D-boxes-0.25-128.log    
fi

#---#
# C #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20C.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20C-boxes-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80C.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80C-boxes-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320C.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320C-boxes-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280C.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280C-boxes-0.25-128.log    
else
    timeout 46800 $@ tests/census/census.txt tests/census/20C.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20C-boxes-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80C.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80C-boxes-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320C.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320C-boxes-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280C.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280C-boxes-0.25-128.log    
fi

#---#
# B #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20B.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20B-boxes-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80B.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80B-boxes-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320B.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320B-boxes-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280B.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280B-boxes-0.25-128.log    
else
    timeout 46800 $@ tests/census/census.txt tests/census/20B.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20B-boxes-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80B.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80B-boxes-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320B.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320B-boxes-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280B.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280B-boxes-0.25-128.log    
fi

#---#
# A #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20A.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20A-boxes-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80A.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80A-boxes-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320A.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320A-boxes-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280A.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280A-boxes-0.25-128.log    
else
    timeout 46800 $@ tests/census/census.txt tests/census/20A.py --domain boxes --lower 0.25 --upper 2 | tee tests/census/logs2/census-20A-boxes-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80A.py --domain boxes --lower 0.25 --upper 8 | tee tests/census/logs2/census-80A-boxes-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320A.py --domain boxes --lower 0.25 --upper 32 | tee tests/census/logs2/census-320A-boxes-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280A.py --domain boxes --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280A-boxes-0.25-128.log    
fi

#==========#
# symbolic #
#==========#

#---#
# F #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20F.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20F-symbolic-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80F.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80F-symbolic-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320F.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320F-symbolic-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280F.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280F-symbolic-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20F.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20F-symbolic-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80F.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80F-symbolic-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320F.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320F-symbolic-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280F.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280F-symbolic-0.25-128.log
fi

#---#
# E #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20E.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20E-symbolic-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80E.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80E-symbolic-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320E.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320E-symbolic-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280E.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280E-symbolic-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20E.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20E-symbolic-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80E.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80E-symbolic-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320E.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320E-symbolic-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280E.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280E-symbolic-0.25-128.log
fi

#---#
# D #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20D.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20D-symbolic-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80D.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80D-symbolic-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320D.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320D-symbolic-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280D.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280D-symbolic-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20D.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20D-symbolic-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80D.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80D-symbolic-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320D.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320D-symbolic-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280D.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280D-symbolic-0.25-128.log
fi

#---#
# C #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20C.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20C-symbolic-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80C.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80C-symbolic-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320C.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320C-symbolic-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280C.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280C-symbolic-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20C.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20C-symbolic-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80C.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80C-symbolic-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320C.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320C-symbolic-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280C.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280C-symbolic-0.25-128.log
fi

#---#
# B #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20B.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20B-symbolic-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80B.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80B-symbolic-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320B.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320B-symbolic-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280B.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280B-symbolic-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20B.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20B-symbolic-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80B.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80B-symbolic-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320B.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320B-symbolic-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280B.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280B-symbolic-0.25-128.log
fi

#---#
# A #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20A.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20A-symbolic-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80A.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80A-symbolic-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320A.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320A-symbolic-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280A.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280A-symbolic-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20A.py --domain symbolic --lower 0.25 --upper 2 | tee tests/census/logs2/census-20A-symbolic-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80A.py --domain symbolic --lower 0.25 --upper 8 | tee tests/census/logs2/census-80A-symbolic-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320A.py --domain symbolic --lower 0.25 --upper 32 | tee tests/census/logs2/census-320A-symbolic-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280A.py --domain symbolic --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280A-symbolic-0.25-128.log
fi

#==========#
# deeppoly #
#==========#

#---#
# F #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20F.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20F-deeppoly-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80F.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80F-deeppoly-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320F.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320F-deeppoly-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280F.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280F-deeppoly-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20F.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20F-deeppoly-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80F.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80F-deeppoly-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320F.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320F-deeppoly-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280F.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280F-deeppoly-0.25-128.log
fi

#---#
# E #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20E.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20E-deeppoly-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80E.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80E-deeppoly-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320E.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320E-deeppoly-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280E.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280E-deeppoly-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20E.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20E-deeppoly-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80E.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80E-deeppoly-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320E.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320E-deeppoly-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280E.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280E-deeppoly-0.25-128.log
fi

#---#
# D #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20D.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20D-deeppoly-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80D.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80D-deeppoly-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320D.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320D-deeppoly-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280D.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280D-deeppoly-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20D.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20D-deeppoly-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80D.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80D-deeppoly-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320D.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320D-deeppoly-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280D.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280D-deeppoly-0.25-128.log
fi

#---#
# C #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20C.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20C-deeppoly-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80C.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80C-deeppoly-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320C.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320C-deeppoly-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280C.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280C-deeppoly-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20C.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20C-deeppoly-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80C.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80C-deeppoly-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320C.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320C-deeppoly-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280C.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280C-deeppoly-0.25-128.log
fi

#---#
# B #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20B.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20B-deeppoly-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80B.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80B-deeppoly-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320B.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320B-deeppoly-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280B.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280B-deeppoly-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20B.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20B-deeppoly-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80B.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80B-deeppoly-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320B.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320B-deeppoly-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280B.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280B-deeppoly-0.25-128.log
fi

#---#
# A #
#---#

if [ "$(uname)" == "Darwin" ]; then
    gtimeout 46800 $@ tests/census/census.txt tests/census/20A.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20A-deeppoly-0.25-2.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/80A.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80A-deeppoly-0.25-8.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/320A.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320A-deeppoly-0.25-32.log
    gtimeout 46800 $@ tests/census/census.txt tests/census/1280A.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280A-deeppoly-0.25-128.log
else
    timeout 46800 $@ tests/census/census.txt tests/census/20A.py --domain deeppoly --lower 0.25 --upper 2 | tee tests/census/logs2/census-20A-deeppoly-0.25-2.log
    timeout 46800 $@ tests/census/census.txt tests/census/80A.py --domain deeppoly --lower 0.25 --upper 8 | tee tests/census/logs2/census-80A-deeppoly-0.25-8.log
    timeout 46800 $@ tests/census/census.txt tests/census/320A.py --domain deeppoly --lower 0.25 --upper 32 | tee tests/census/logs2/census-320A-deeppoly-0.25-32.log
    timeout 46800 $@ tests/census/census.txt tests/census/1280A.py --domain deeppoly --lower 0.25 --upper 128 | tee tests/census/logs2/census-1280A-deeppoly-0.25-128.log
fi
