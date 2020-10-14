#!/usr/bin/env bash
source /data/bossapp/.bash_profile
uuid=`/usr/bin/uuidgen`
echo ${uuid}
uuid2=${uuid//-/}
echo ${uuid2}
echo ${uuid2} > ./unique_name
#pinoctl start --force --unique_name=${uuid2} --email=xxx@jd.com --jenkinsUser=${erp} --group=${your group} --runtime={you runtime}  --PinoAI=.  .
echo "excute cmd: 9nctl start --force --unique_name=${uuid2} --group=ads-sz-dpa-model ."
#9nctl resources --cluster=langfang
9nctl start --force --unique_name=${uuid2} --group=ads-sz-dpa-model .
