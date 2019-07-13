#!/bin/bash
function loading() {
  wheel=("-" "\\" "|" "/" "-")
  for i in ${wheel[*]}
    do 
      printf "[\r$i] $1"; 
      sleep 0.1;
    done;
}
function wait_for_process() {
  while [[ `ps aux | grep $1 | wc -l` -le 1 ]]
  do
    loading $2
  done
  wait $1
  return "$?"
}
function report_error(){
[[ $1 -ne 0 ]] && printf "\r\033[K[X] Error occurred! Check $2" && exit 1 || rm $2
}

# Script Starts Here
pushd $HOME

pip3 -q install -U tfds-nightly &> pip.log &
wait_for_process $! "Installing Tensorflow Datasets"
report_error $? "pip.log"
printf "\r\033[K[-] Installed Tensorflow Datasets\n"
pip3 -q install -U tf-hub-nightly &> pip.log &
wait_for_process $! "Installing Tensorflow Hub"
report_error $? "pip.log"
printf "\r\033[K[-] Installed Tensorflow Hub\n"

DATASET="gs://images.cocodataset.org/train2014"
EXTRACTDIR="$HOME/coco2014/train/none"
CODE="import os;
  import tensorflow_datasets as tfds;
  dl_config = tfds.download.DownloadConfig(manual_dir=os.path.expanduser('~/datadir'));
  ds = tfds.load('image_label_folder/dataset_name=coco2014',
    split='train',
    as_supervised=True,
    download_and_prepare_kwargs={'download_config':dl_config})"

CODE=$(echo $CODE | sed -r s/"[[:space:]]*(;|,|=)[[:space:]]*"/"\1"/g)

if [ ! -d "datadir" ]
then
	mkdir -p datadir/coco2014/train/none
  printf " [*] Downloading and Extracting Images from: $DATASET"
  gsutil -m rsync $DATASET $EXTRACTDIR &>download.log &
  wait_for_process $! "Downloading and Extracting Images from: $DATASET"
  report_error $? "download.log"
  printf "\r\033[K[-] Done Downloading and Extracting!\n" 
  
  python3 -c "$CODE" &>parse.log &
  wait_for_process $! "Parsing Dataset to TF Records."
	report_error $? "parse.log"
  printf "\r\033[K[-] Done Parsing to TF Records\n"
fi
# Creating Log and  Model Dump Directories"
[ ! -d "logdir" ] && mkdir logdir
[ ! -d "modeldir" ] && mkdir modeldir

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
echo "[-] Log of trainig code: $HOME/logdir/main.log"
echo "[-] Logs of tensorboard: $HOME/logdir/access.log"
printf "PIDs of created jobs\n"
echo "[-] PID Tensorboard (root): "${TB_PID:1}
PY_PID=$(pgrep -u `whoami` -f python3)
PY_PID=$(printf ",%s" "${PY_PID[@]}")
echo "[-] PID Python3 ($(whoami)): "${PY_PID:1}
