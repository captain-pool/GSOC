#!/bin/bash
USERHOME="/home/$(who am i | awk '{print $1}')"
function download_flickr100k(){
  [[ ! -d "$USERHOME/datadir/downloads/flickr100k" ]] && mkdir -p "$USERHOME/datadir/downloads/flickr100k"
  [[ ! -d "$USERHOME/datadir/flickr100k/train/none" ]] && mkdir -p "$USERHOME/datadir/flickr100k/train/none"
  for i in `seq 1 40`; do
      printf "Downloading %.2d out of 40\n" $i;
      partNum=$(printf "%.2d" $i)
      cmd="wget -P $USERHOME/datadir/downloads/flickr100k/ http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/ox100k/oxc1_100k.part$partNum.rar";
      [[ -f "$USERHOME/datadir/downloads/flickr100k/oxc1_100k.part$partNum.rar" ]] && continue
      echo 'Executing: '$cmd
      $cmd
  done

  if [[ `ls "$USERHOME/datadir/flickr100k/train/none" | wc -l` -lt 90000 ]]
  then
    unrar e "$USERHOME/datadir/downloads/flickr100k/oxc1_100k.part40.rar" "$USERHOME/datadir/flickr100k/train/none/"
  fi
  rm -f "$USERHOME/datadir/flickr100k/train/none/portrait_000801.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/portrait_000801.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/portrait_000801.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/festival_001907.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/green_002662.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/green_002662.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/baby_001317.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/school_001535.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/cat_001894.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/park_001319.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/autumn_000335.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/06_000443.jpg"
  rm -f "$USERHOME/datadir/flickr100k/train/none/portrait_000801.jpg"
}

download_flickr100k

mkdir -p "$USERHOME/logdir"
mkdir -p "$USERHOME/modeldir"

printf "\r\033[K Checking Tensorflow Installation ... "
python3 -u -c "import tensorflow" &> error.log &

wait $!
[[ $? -ne 0 ]] && printf "\r\033[K\033[1m Error Importing Tensorflow. Check $PWD/error.log\033[0m" && exit 1 || rm -f error.log
printf "\n"
printf "\033[1mPInstalling Tensorflow Datasets\033[0m\n"
pip3 install tfds-nightly
printf "\033[1mPInstalling Tensorflow Hub\033[0m\n"
pip3 install tf-hub-nightly
git clone https://github.com/captain-pool/GSOC

pushd $PWD/GSOC/E2_ESRGAN
python3 -u main.py --data_dir "$USERHOME/datadir" \
	--model_dir "$USERHOME/modeldir" \
	--manual --log_dir "$USERHOME/logdir" \
--phases "phase1_phase2" -vvv&>"$USERHOME/logdir/main.log" &
