import torch
import sys

sys.path.append("..")

from deepsplit.admm import *
from mnist_nn_models import mnist_fc, mnist_loaders
import matplotlib.pyplot as plt

torch.set_grad_enabled(False)

if __name__ == '__main__':
    model_param_name = 'mnist_fc.pth'
    nn_model = mnist_fc()
    nn_model.load_state_dict(torch.load(model_param_name))
    nn_model.cuda()

    # load test data set
    batch_size = 1
    train_loader, test_loader = mnist_loaders(batch_size)

    data_loader = test_loader

    eps = 0.01
    layer_list = list(nn_model)[1:]
    data_iter = iter(data_loader)
    data = data_iter.next()
    image, label = data[0].cuda(), data[1].cuda()

    x_input = nn_model[0](image)
    lb_input = x_input - eps
    ub_input = x_input + eps

    # find predicted class
    score = nn_model(x_input)
    _, class_num = torch.max(score, dim=1)

    num_samples = x_input.size(0)
    num_classes = 10

    class_idx = 1
    c = torch.zeros(num_samples, num_classes)
    c[:, class_idx] = -torch.ones(num_samples)
    for j in range(num_samples):
        c[j, class_num[j]] += 1
    c = c.cuda()

    rho = 1.0
    alg_options = {'rho': rho, 'eps_abs': 1e-4, 'eps_rel': 1e-3, 'residual_balancing': True, 'max_iter': 10000,
                   'record': True, 'verbose': True, 'alpha': 1.6}

    init_module = InitModule(layer_list, x_input, lb_input, ub_input, pre_act_bds_list=None)
    admm_module = init_module.init_ADMM_module()
    admm_sess = ADMM_Session([admm_module], lb_input, ub_input, c, rho)
    objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)

    print('')
    plt.figure()
    obj_values = [item.item() for item in result['obj_list']]
    plt.plot(obj_values, label = 'objective')
    plt.legend()
    plt.xlabel('iter')
    plt.ylabel('obj')

    plt.figure()
    rp_values = [item.item() for item in result['rp_list']]
    primal_tol_values = [item.item() for item in result['p_tol_list']]
    plt.plot(rp_values, label = 'primal res.')
    plt.plot(primal_tol_values,label = 'primal tol.')
    plt.xlabel('iter')
    plt.legend()

    plt.figure()
    rd_values = [item.item() for item in result['rd_list']]
    dual_tol_values = [item.item() for item in result['d_tol_list']]
    plt.plot(rd_values, label='primal res.')
    plt.plot(dual_tol_values, label='primal tol.')
    plt.xlabel('iter')
    plt.legend()

    plt.show()
