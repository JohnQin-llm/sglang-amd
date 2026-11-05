#!/bin/bash
# loopmix.sh file.amdgcn [n]: instruction mix of the n-th (default 1) loop, header .. back-edge
f=$1; n=${2:-1}
H=$(grep -n "Loop Header" $f | sed -n ${n}p)
L=${H%%:*}; lab=$(echo "$H" | sed 's/^[0-9]*:\(\.LBB[0-9_]*\):.*/\1/')
E=$(grep -n "s_cbranch.* $lab\$\|s_branch $lab\$" $f | tail -1 | cut -d: -f1)
echo "loop $lab lines $L-$E"
python3 /data/muse455/attn2/tools/isa_mix.py $f $L $E
