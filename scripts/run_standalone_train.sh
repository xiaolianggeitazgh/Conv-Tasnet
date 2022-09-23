
if [ $# != 1 ]
then
  echo "==========================================================================="
  echo "Please run the script as: "
  echo "For example:"
  echo "bash run_standalone_train.sh [DEVICE_ID]  [DATA_DIR] [FILE_LIST] [BATCHSIZE] "
  echo "bash run_standalone_train.sh 0   dataset/vimeo_septuplet/sequencesdataset/vimeo_septuplet/sep_trainlist.txt  4 "
  echo "Using absolute path is recommended"
  echo "==========================================================================="
  exit 1
fi

export DEVICE_ID=$1
export RANK_ID=0
export RANK_SIZE=1
export SLOG_PRINT_TO_STDOUT=0


rm -rf ./train_conv_tasnet
mkdir ./train_conv_tasnet
cp -r ../src ./train_conv_tasnet
cp -r ../*.py ./train_conv_tasnet
cd ./train_conv_tasnet || exit
python train.py --device_id=$DEVICE_ID  > train.log 2>&1 &
