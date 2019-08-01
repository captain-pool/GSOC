#! /bin/bash
# $BASE contains the path to the cloud bucket.
# $TPU_NAME contains the name of the cloud bucket.
gsutil -m rm -r $BASE/distillation/logdir/** &>/dev/null
sudo tensorboard --logdir $BASE/distillation/logdir --port 80 &> $HOME/logdir/access.log &
python3 main.py --logdir $BASE/distillation/logdir \
	--datadir $BASE/datadir/coco2014_esrgan\
	--tpu $TPU_NAME \
	--modeldir $BASE/distillation/modeldir \
	--type "comparative" -vvv &> $HOME/logdir/main.log &

