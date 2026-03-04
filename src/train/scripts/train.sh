# python3 -m scripts.train_recon --config config/cifar10_1.yaml --dataset cifar10
# python3 -m scripts.train_recon --config config/cifar10_2.yaml --dataset cifar10

# python3 -m scripts.train_recon --config config/cifar10_3_1.yaml --dataset cifar10
# python3 -m scripts.train_recon --config config/cifar10_3_2.yaml --dataset cifar10

# python3 -m scripts.train_recon --config config/cifar10_4_1.yaml --dataset cifar10
# python3 -m scripts.train_recon --config config/cifar10_4_2.yaml --dataset cifar10

# python3 -m scripts.train_classification --batch_size 128 --max_epoch 2000 --model resnet --output_folder resnet --output_name resnet3 --saver

# python3 -m scripts.train_classification --batch_size 128 --max_epoch 2000 --model vit --output_folder vit --output_name vit_medium --saver

# python3 -m scripts.train_mnist --dataset torch/mnist --batch_size 128 --max_epoch 20 --model fc --output_folder fc --saver --output_name  mnist_small

# python3 -m example.scripts.generate_instances_mnistfc --model_type fc --model_name mnist_small --eps 0.1
# python3 -m example.scripts.filter_instances --device cpu --model_name mnist_small --eps 0.1
# python3 -m example.scripts.extract_filtered_instances --model_name mnist_small --eps 0.1

# python3 -m example.scripts.filter_instances --device cuda --model_name mnist_256x2 --eps 0.15
# python3 -m example.scripts.filter_instances --device cuda --model_name mnist_256x3 --eps 0.15
# python3 -m example.scripts.filter_instances --device cuda --model_name mnist_256x6 --eps 0.08


# python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x2 --eps 0.15
# python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x3 --eps 0.15
# python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x4 --eps 0.12
# python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x5 --eps 0.08
# python3 -m example.scripts.extract_filtered_instances --model_name mnist_256x6 --eps 0.08


# RNN
python3 -m train.scripts.train_rnn_mnist --batch_size 128 --max_epoch 20 --model lstm --output_folder lstm --saver --output_name  mnist_lstm_128x2 --device cuda
python3 -m train.scripts.train_rnn_mnist --batch_size 128 --max_epoch 20 --model lstm --output_folder lstm --saver --output_name  mnist_lstm_64x1 --device cuda

python3 -m train.scripts.train_rnn_mnist --batch_size 128 --max_epoch 20 --model gru --output_folder gru --saver --output_name  mnist_gru_128x2 --device cuda
python3 -m train.scripts.train_rnn_mnist --batch_size 128 --max_epoch 20 --model gru --output_folder gru --saver --output_name  mnist_gru_64x1 --device cuda

# Generate instances
python3 -m train.scripts.generate_instances_mnist_rnn --model_name mnist_gru_64x1  --model_type gru --test --eps 0.05 --num_spec 100 --num_perturb 784
python3 -m train.scripts.generate_instances_mnist_rnn --model_name mnist_gru_128x2 --model_type gru --test --eps 0.05 --num_spec 100 --num_perturb 784

python3 -m train.scripts.generate_instances_mnist_rnn --model_name mnist_lstm_64x1  --model_type lstm --test --eps 0.05 --num_spec 100 --num_perturb 784
python3 -m train.scripts.generate_instances_mnist_rnn --model_name mnist_lstm_128x2 --model_type lstm --test --eps 0.05 --num_spec 100 --num_perturb 784
