lockloc="locks/IM_LOCK"
while [ ! -f "$lockloc" ]; do
    echo 'Lock not found, waiting 1 second'
    sleep 1
done

echo 'Lock found! Taking picture...'
raspistill -o "imgs/tmp.jpg"
