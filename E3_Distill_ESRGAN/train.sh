#! /bin/bash
# $BASE contains the path to the cloud bucket.
# $TPU_NAME contains the name of the cloud bucket.
sudo tensorboard --logdir $BASE/logdir --port 80 &> $HOME/logdir/access.log &
python3 main.py --logdir $BASE/distillation/logdir \
	--datadir $BASE/datadir/coco2014_esrgan\
	--tpu $TPU_NAME \
	--modeldir $BASE/distillation/modeldir \
	--type "adversarial" -vvv &> $HOME/logdir/main.log &

