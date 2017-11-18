export PYTHONUNBUFFERED="True"
LOG="log/train-`date +'%Y-%m-%d-%H-%M-%S'`.log"
python train.py --gpu 0 --train_dir /home/code/panhongyu/datasets/mnist --save_dir results/ --config config.yml > $LOG
