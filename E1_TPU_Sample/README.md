# Image Retraining Sample

**Topics:** Tensorflow 2.0, TF Hub, Cloud TPU

## Specs
### Cloud TPU

**TPU Type:** v2.8
**Tensorflow Version:** 1.14

### Cloud VM

**Machine Type:** n1-standard-2
**OS**: Debian 9
**Tensorflow Version**: Came with tf-nightly. Manually installed Tensorflow 2.0 Beta

Launching Instance and VM
---------------------------
- Open Google Cloud Shell
- `ctpu up -tf-version 1.14`
- If cloud bucket is not setup automatically, create a cloud storage bucket
with the same name as TPU and the VM
- enable HTTP traffic for the VM instance
- SSH into the system
  - `pip3 uninstall -y tf-nightly`
  - `pip3 install -r requirements.txt`
  - `export CTPU_NAME=<common name of the tpu, vm and bucket>`


Running Tensorboard:
----------------------
### Pre Requisites
```bash
$ sudo -i
$ pip3 uninstall -y tf-nightly
$ pip3 install tensorflow==2.0.0-beta0
$ exit
```

### Launch
```bash
$ sudo tensorboard --logdir gs://$CTPU_NAME/model_dir --port 80 &>/dev/null &
```
To view Tensorboard, Browse to the Public IP of the VM Instance

Running the Code:
----------------------
#### Train The Model

```bash
$ python3 image_retraining_tpu.py --tpu $CTPU_NAME --use_tpu \
--modeldir gs://$CTPU_NAME/modeldir \
--datadir gs://$CTPU_NAME/datadir \
--logdir gs://$CTPU_NAME/logdir \
--num_steps 2000 \
--dataset horses_or_humans
```
Training Saves one single checkpoint at the end of training. This checkpoint can be loaded up
later to export a SavedModel from it.

#### Export Model

```bash
$ python3 image_retraining_tpu.py --tpu $CTPU_NAME --use_tpu \
--modeldir gs://$CTPU_NAME/modeldir \
--datadir gs://$CTPU_NAME/datadir \
--logdir gs://$CTPU_NAME/logdir \
--dataset horses_or_humans \
--export_only \
--export_path modeldir/model
```
Exporting SavedModel of trained model
----------------------------
The trained model gets saved at `gs://$CTPU_NAME/modeldir/model` by default if the path is not explicitly stated using `--export_path`
