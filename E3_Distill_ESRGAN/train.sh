#! /bin/bash
sudo tensorboard --logdir $HOME/logdir --port 80 &> $HOME/logdir/access.log &
python3 main.py --logdir $HOME/logdir \
	--datadir $HOME/datadir --manual \
	--modeldir $BASE/distillation/modeldir \
	--type "adversarial" -vvv &> $HOME/logdir/main.log &

