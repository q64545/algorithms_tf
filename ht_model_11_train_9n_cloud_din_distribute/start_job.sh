#!/usr/bin/env bash
source /data/bossapp/.bash_profile
uuid=`/usr/bin/uuidgen`
echo ${uuid}
uuid2=${uuid//-/}
echo ${uuid2}
echo ${uuid2} > /export/App/9n_online_dir/datahouse/python/biz/shop_dpa/ht_model_11_train_9n_cloud_din_distribute/unique_name
#project_path=`cat fpwd`
#pinoctl start --force --unique_name=${uuid2} --email=xxx@jd.com --jenkinsUser=${erp} --group=${your group} --runtime={you runtime}  --PinoAI=.  .
echo "excute cmd: 9nctl start --force --unique_name=${uuid2} --group=ads-sz-dpa-model /export/App/9n_online_dir/datahouse/python/biz/shop_dpa/ht_model_11_train_9n_cloud_din_distribute"
#9nctl resources --cluster=langfang
9nctl start --force --unique_name=${uuid2} --group=ads-sz-dpa-model /export/App/9n_online_dir/datahouse/python/biz/shop_dpa/ht_model_11_train_9n_cloud_din_distribute

if [ "$?" != "0" ];then
    echo 'start train error!'
    exit 1
fi

echo "task ${uuid2} is submitted..."

function on_fail()
{
    echo "Fail"
    exit 1
}
function on_transfered()
{
    echo "Transfered"
    exit 0
}
function on_success()
{
    echo "Success"
    exit 0
}
function wait_released()
{
    local _count="1"
    while [ "$_count" != "0" ];
    do
        _count=`pinoctl list|grep ${uuid2}|wc -l`
        sleep 10
    done
}
function on_exit()
{
    pinoctl clean ${uuid2}
    wait_released
    echo "Exit"
}
trap on_exit EXIT
result="Running"
while true ; do
    result=`pinoctl status ${uuid2}`
    if [ "${result}" == "Fail" ]; then
        on_fail
    elif [ "${result}" == "Transfered" ]; then
        on_transfered
    elif [ "${result}" == "Success" ]; then
        on_success
    elif [ "${result}" == "" ]; then
    	on_fail
    elif [ "${result}" == "not found" ]; then
        echo "no resouces, submit fail"
    	on_fail
    else
        echo "Wait Model ..."
        sleep 5s
    fi
done
count=0
#while true ; do
#    ((count++))
#    result=`pinoctl list|grep ${uuid2}|awk '{print $4}'`
#    if [ "${result}" == "Transfered" ] || [ ${count} -gt 720 ] ; then
#        on_transfered
#    else
#        echo "Waiting Transfer Model ..."
#        sleep 1m
#    fi
#done