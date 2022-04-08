conda create -y -n oval21-generation python=3.6
conda activate oval21-generation
conda install -y pytorch torchvision cudatoolkit=9.2 -c pytorch
pip install .