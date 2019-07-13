#!/bin/bash

pushd $HOME
DATASET="gs://images.cocodataset.org/train2014"
EXTRACTDIR="$HOME/coco2014/train/none"
CODE="import os;import tensorflow_datasets as tfds;dl_config = tfds.download.DownloadConfig(manual_dir=os.path.expanduser('~/datadir'));ds = tfds.load('image_label_folder/dataset_name=coco2014', split='train', as_supervised=True, download_and_prepare_kwargs={'download_config':dl_config})"
if [ ! -d "datadir" ]
then
	mkdir -p datadir/coco2014/train/none
	gsutil -m rsync $DATASET $EXTRACTDIR
	python3 -c "$CODE"
	[[ $? -ne 0 ]] && echo "Error Parsing to TF Records" && exit 1

fi

if [ ! -d "logdir" ]
then
	mkdir logdir
fi
if [ ! -d "modeldir" ]
then
	mkdir modeldir
fi

if [ $(ps aux|grep tensorboard|wc -l) -le 1 ]
then
	sudo tensorboard --logdir $HOME/logdir \
		--port 80 &> $HOME/logdir/access.log &
fi

if [ ! -d "GSOC" ]
then
	git clone https://captain-pool/GSOC
fi
pushd $HOME/GSOC/E2_ESRGAN
rm -rf cache/*.lockfile
python3 main.py --data_dir $HOME/datadir \
	--model_dir $HOME/modeldir \
	--manual --log_dir $HOME/logdir \
	--phases phase2 -vvv&>$HOME/logdir/main.log &
popd
TB_PID=$(IFS=' ' pgrep -u root -f tensorboard)
TB_PID=$(printf ",%s" "${TB_PID[@]}")
echo "PID Tensorboard (root): "${TB_PID:1}
PY_PID=$(pgrep -u `whoami` -f python3)
PY_PID=$(printf ",%s" "${PY_PID[@]}")
echo "PID Python3 ($(whoami)): "${PY_PID:1}
