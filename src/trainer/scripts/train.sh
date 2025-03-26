python3 -m scripts.train_recon --config config/cifar10_1.yaml --dataset cifar10
python3 -m scripts.train_recon --config config/cifar10_2.yaml --dataset cifar10

python3 -m scripts.train_recon --config config/cifar10_3_1.yaml --dataset cifar10
python3 -m scripts.train_recon --config config/cifar10_3_2.yaml --dataset cifar10

python3 -m scripts.train_recon --config config/cifar10_4_1.yaml --dataset cifar10
python3 -m scripts.train_recon --config config/cifar10_4_2.yaml --dataset cifar10

python3 -m scripts.train_classification --dataset torch/cifar10 --batch_size 128 --max_epoch 200 --model resnet --output_folder resnet --output_name resnet3 --saver


python3 -m scripts.train_mnist --dataset torch/mnist --batch_size 128 --max_epoch 20 --model fc --output_folder fc --saver --output_name  mnist_small

python3 -m example.scripts.generate_instances_mnistfc --model_type fc --model_name mnist_small --eps 0.1
python3 -m example.scripts.filter_instances --device cpu --model_name mnist_small --eps 0.1
python3 -m example.scripts.extract_filtered_instances --model_name mnist_small --eps 0.1

python3 -m example.scripts.filter_instances --device cuda --model_name mnist_256x2 --eps 0.15
python3 -m example.scripts.filter_instances --device cuda --model_name mnist_256x3 --eps 0.15
python3 -m example.scripts.filter_instances --device cuda --model_name mnist_256x6 --eps 0.08


python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x2 --eps 0.15
python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x3 --eps 0.15
python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x4 --eps 0.12
python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x5 --eps 0.08
python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x6 --eps 0.08