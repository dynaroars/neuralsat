import argparse
import torch
import time
from networks import FullyConnected, Conv
from abstract_nets import AbstractFullyConnected, AbstractConv, AbstractRelu
import signal
from contextlib import contextmanager
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# Warning we get: (torch 1.7 on) .../torch/autograd/__init__.py:132: UserWarning: CUDA initialization: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx (Triggered internally at  /pytorch/c10/cuda/CUDAFunctions.cpp:100.

DEVICE = 'cpu'
INPUT_SIZE = 28
NUM_EPOCHS = 100 # number insanely high to make the code loop
LEARNING_RATE = 3
MOMENTUM = 0.99
MAX_TIME = 180
GAMMA = 0.95

class LamdaLoss(torch.nn.Module):
    """ Custom loss function to optimise the lamdas"""
    def __init__(self):
        super(LamdaLoss, self).__init__()

    @staticmethod
    def forward(last_low, last_high, right_class):
        assert (last_high >= last_low).all(), "Error with the box bounds: low>high"

        unit = torch.ones_like(last_low)
        zero = torch.zeros_like(last_low)
        loss_out = torch.sum(torch.maximum(zero,(-1*last_low)))
        return loss_out


class LamdaOptimiser():
    """ class that contains all the optimisation logic for the lamdas"""
    def __init__(self, inputs, low_orig, high_orig, net, true_label, start_time):
        self._inputs = inputs
        self._low_orig = low_orig
        self._high_orig = high_orig
        self._net = net
        self._true_label = true_label
        self._start_time = start_time
        self.loss = LamdaLoss().to(DEVICE)
        self.lr = LEARNING_RATE


    def get_lamdas(self):
        # extract all the lamdas from the net
        self.lamdas = []
        for layer in self._net.layers:
            if type(layer) == AbstractRelu:
                self.lamdas += [layer.lamda]

    def initialise_lamdas_get_out(self):
        # extract all the lamdas from the net

        for layer in self._net.layers:
            if type(layer) == AbstractRelu:
                layer.initialise_lamda(random_lamdas=torch.rand_like(layer.lamda))

    def update_lamdas(self, backsub_order=None, epoch=0):
        """ Wrapping function for all the operation necessary
        to make an optimization step for the lamdas"""

        outputs, low, high = self._net(self._inputs, self._low_orig, self._high_orig)
        # get even tighter bounds
        low, high = self._net.back_sub(true_label=self._true_label, order=backsub_order)
        self.get_lamdas()  # now we know which ones are crossing or not
        self.lr = max(0.5,self.lr*GAMMA)
        optimizer = torch.optim.SGD(self.lamdas, lr=self.lr, momentum=MOMENTUM)
        loss_value = self.loss(low, high, self._true_label)
        optimizer.zero_grad()
        loss_value.backward()
        '''INCLUDING THE FOLLOWING BUT CAN REMOVE OR COMMENT OUT DEPENDING ON FUTURE USE
        new_lamdas = []
        for lamda in self.lamdas:
            if epoch == 0:
                new_lamda = lamda - LEARNING_RATE * lamda.grad.sign()
            elif epoch == 1:
                new_lamda = lamda - 0.5 * lamda.grad.sign()
            else:
                new_lamda = lamda - max(0.2, (1 / epoch)) * lamda.grad

            new_lamdas += [new_lamda]
        '''
        optimizer.step()
        # note: the lamdas have to be between 0 and 1, hence we
        # cannot simply update them with a gradient descent step
        # - we also need to project back to the [0, 1] box
        self._net.clamp_lamdas() # this will make lamda a list again
        #self.get_lamdas()
        return outputs, low, high

    def optimise(self):
        """ Main optimisation loop"""
        verified = False
        epoch = 0
        self._net.set_optimising_relu()
        while not verified:

            print("Epoch " + str(epoch))
            backsub_order = None # I initialise it here because we may want to add a per epoch logic for this
            outputs, low, high = self.update_lamdas(backsub_order, epoch = epoch)
            #         low, high = self._net.back_sub(true_label=self._true_label, order=backsub_order)
            verified = (low.detach().numpy() > 0).all()
            pred_label = outputs.max(dim=0)[1].item()
            assert pred_label == self._true_label #check that only the lamdas have been changed.

            epoch +=1
            end = time.time()
            print("Time: " + str(round(end - self._start_time, 3)))
            if (round(end - self._start_time, 3) > MAX_TIME):  # we're going to go over the limit with next iteration
                print("Timeout!")
                break

        return verified


def prepare_input_verifier(inputs, eps):
    """ 
    This function computes the input to the verifier. 
    Given an input image and a noise level it computes the range 
    of values for every pixel. 
    From the task description: 
    'Note that images have pixel intensities between 0 and 1, 
    e.g. if perturbation is 0.2 and pixel has value 0.9 then you only 
    have to verify range [0.7, 1.0] for intensities of this pixel, 
    instead of [0.7, 1.1]'
    
    """
    # inputs has shape (1,1,28,28)
    # hence also eps has the same shape
    low = torch.max(inputs - eps, torch.tensor(0.0)).to(DEVICE) # may we should limit this to something very small instead than 0?
    high = torch.min(inputs + eps, torch.tensor(1.0)).to(DEVICE)
    return inputs, low, high

def analyze(net, inputs, eps, true_label):
    """
        This function should run the DeepPoly relaxation on the L infinity 
        ball of radius epsilon around the input and verify whether the net 
        would always output the right label. 

        [input +-eps] --> [y_true-label > y_i] for all i != true-label

        Arguments
        ---------
        net: (nn.Module) - either instance of AbstractFullyConnected or AbstractConv  
        inputs: (FloatTensor) - shape (1, 1, 28, 28)
        eps: (float) - noise level
        true_label: (int) - label from 0 to 9  

        Returns
        --------
        (bool) - True if the property is verified
    """
    start = time.time()
    # 1. Define the input box - the format should be defined by us 
    # as it will be used by our propagation function. 
    inputs, low_orig, high_orig = prepare_input_verifier(inputs, eps)
    with torch.no_grad():
        # 2. Propagate the region across the net
        outputs, low, high = net(inputs, low_orig, high_orig)
    pred_label = outputs.max(dim=0)[1].item()
    assert pred_label == true_label
    # 3. Verify the property
    verified = sum((low[true_label]>high).int())==9
    end = time.time()
    print("Propagation done. Time : "+str(round(end-start,3)))
    if verified: return verified
    # 4. Backsubstitute if the property is not verified, otherwise return
    backsub_order = None
    with torch.no_grad():
        low, high = net.back_sub(true_label=true_label, order=backsub_order)
    # for the property to be verified we want all the entries of (y_true - y_j) to be positive
    verified = (low.detach().numpy() > 0).all()
    end = time.time()
    print("Backsubstitution done. Time : " + str(round(end - start, 3)))
    if verified: return verified
    # 5. Update the lamdas to optimise our loss and try to verify again
    print("Optimising the lamdas...")
    lamda_optimiser = LamdaOptimiser(inputs, low_orig, high_orig, net, true_label, start)
    verified = lamda_optimiser.optimise()
    return verified


def main():
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args()

    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])


    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
        abstract_net = AbstractFullyConnected(DEVICE, INPUT_SIZE,[100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
        abstract_net = AbstractConv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
        abstract_net = AbstractConv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
        abstract_net = AbstractConv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False


    # here we are loading the pre-trained net weights 
    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))
    abstract_net.load_weights(net)
    if type(abstract_net)==AbstractConv:
        #cast it to fully connected
        print("Converting the network to fully connected...")
        abstract_net = abstract_net.make_fully_connected(DEVICE).to(DEVICE)
        print("Conversion done")

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    if analyze(abstract_net, inputs, eps, true_label):
        print('verified')
    else:
        print('not verified')

    print("-"*20)


if __name__ == '__main__':
    main()