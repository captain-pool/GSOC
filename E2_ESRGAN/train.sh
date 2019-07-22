#!/bin/bash
function loading() {
  wheel=("-" "\\" "|" "/")
  for i in ${wheel[*]}
    do 
      printf "\r[$i] $1"; 
      sleep 0.1;
    done;
}
function wait_for_process() {
  while [[ `ps aux | grep $1 | wc -l` -gt 1 ]]
  do
    loading "$2"
  done
  wait $1
  return "$?"
}
function report_error(){
[[ $1 -ne 0 ]] && printf "\r\033[K[X] Error occurred! Check $2" && exit 1 || rm -f "$2"
}

# Script Starts Here

# Param 1: Base Directory
# Param 2: Phases to Work on
# Param 3: coco2014 / Extract Location of the Dataset

[[ ${#1} -gt 0 ]] && BASEDIR="$1" || BASEDIR="$HOME"
[[ ${#2} -gt 0 ]] && PHASES="$2" || PHASES="phase1_phase2"
[[ "${3,,}" == "coco2014" ]] || EXTRACTDIR="$3"

python3 -c "import tensorflow" &> /dev/null &
wait_for_process $! "Checking Tensorflow Installation"
report_error $? "Tensorflow Installation"

pushd "$BASEDIR"
python3 -c "import tensorflow_datasets" &>/dev/null &
wait_for_process $! "Checking TFDS Installation"
if [[ $? -ne 0 ]]
then
	pip3 -q install -U tfds-nightly &> pip.log &
	wait_for_process $! "Installing Tensorflow Datasets"
	report_error $? "pip.log"
	printf "\r\033[K[-] Installed Tensorflow Datasets\n"
fi

python3 -c "import tensorflow_hub" &>/dev/null &
wait_for_process $! "Checking TF-Hub Installation"
if [[ $? -ne 0 ]]
then
	pip3 -q install -U tf-hub-nightly &> pip.log &
	wait_for_process $! "Installing Tensorflow Hub"
	report_error $? "pip.log"
	printf "\r\033[K[-] Installed Tensorflow Hub\n"
fi

if [[ ${#EXTRACTDIR} -eq 0 ]]
then
  DATASET="gs://images.cocodataset.org/train2014"
  EXTRACTDIR="$BASEDIR/datadir/coco2014/train/none"
fi
IFS=/ read -ra ds <<<"$EXTRACTDIR"
DATADIR=$(dirname $(dirname $(dirname $EXTRACTDIR)))
CODE="import os;
  import tensorflow_datasets as tfds;
  dl_config = tfds.download.DownloadConfig(manual_dir=os.path.expanduser('$DATADIR'));
  ds = tfds.load('image_label_folder/dataset_name=${ds[-3],,}',
    split='train',
    as_supervised=True,
    download_and_prepare_kwargs={'download_config':dl_config});"

CODE=$(echo $CODE | sed -r s/"[[:space:]]*(;|,|=)[[:space:]]*"/"\1"/g)

if [ ! -d "$EXTRACTDIR" ]
then
  if [[ ${#DATASET} -gt 0 ]]
  then
    mkdir -p "$EXTRACTDIR"
    printf " [*] Downloading and Extracting Images from: $DATASET"
    gsutil -m rsync $DATASET $EXTRACTDIR &>download.log &
    wait_for_process $! "Downloading and Extracting Images from: $DATASET"
    report_error $? "download.log"
    printf "\r\033[K[-] Done Downloading and Extracting.\n" 
  else
    echo "Data Directory Doesn't Exist! Exiting..." && exit 1
  fi
fi
python3 -u -c "$CODE" &>parse.log &
wait_for_process $! "Parsing Dataset to TF Records."
report_error $? "parse.log"
printf "\r\033[K[-] Done Parsing to TF Records\n"

# Creating Log and  Model Dump Directories"
[ ! -d "logdir" ] && mkdir logdir
[ ! -d "modeldir" ] && mkdir modeldir

if [ $(ps aux|grep tensorboard|wc -l) -le 1 ]
then
	sudo tensorboard --logdir "$BASEDIR/logdir" \
		--port 80 &> "$BASEDIR/logdir/access.log" &
fi

if [ ! -d "GSOC" ]
then
	git clone https://captain-pool/GSOC
fi
pushd "$BASEDIR/GSOC/E2_ESRGAN"

rm -f cache/*.lockfile

python3 -u main.py --data_dir "$DATADIR" \
	--model_dir "$BASEDIR/modeldir" \
	--manual --log_dir "$BASEDIR/logdir" \
	--phases $PHASES -vvv&>"$BASEDIR/logdir/main.log" &
popd

TB_PID=$(IFS=' ' pgrep -u root -f tensorboard)
TB_PID=$(printf ",%s" "${TB_PID[@]}")
echo "[-] Log of training code: $BASEDIR/logdir/main.log"
echo "[-] Logs of tensorboard: $BASEDIR/logdir/access.log"
printf "\033[1mPIDs of created jobs\033[0m\n"
echo "[-] PID Tensorboard (root): "${TB_PID:1}
PY_PID=$(pgrep -u `whoami` -f python3)
PY_PID=$(printf ",%s" "${PY_PID[@]}")
echo "[-] PID Python3 ($(whoami)): "${PY_PID:1}
