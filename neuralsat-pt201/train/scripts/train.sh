python3 -m scripts.train_recon --config config/cifar10_1.yaml --dataset cifar10
python3 -m scripts.train_recon --config config/cifar10_2.yaml --dataset cifar10

python3 -m scripts.train_recon --config config/cifar10_3_1.yaml --dataset cifar10
python3 -m scripts.train_recon --config config/cifar10_3_2.yaml --dataset cifar10

python3 -m scripts.train_recon --config config/cifar10_4_1.yaml --dataset cifar10
python3 -m scripts.train_recon --config config/cifar10_4_2.yaml --dataset cifar10

python3 -m scripts.train_classification --dataset torch/cifar10 --batch_size 128 --max_epoch 200 --model resnet --output_folder resnet --output_name resnet3 --saver
