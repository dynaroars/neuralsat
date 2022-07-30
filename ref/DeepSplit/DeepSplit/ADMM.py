
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.simplefilter("always")

import time
import cvxpy as cp
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
# torch.set_grad_enabled(False)


# =============================================================================
# ADMM layer
# =============================================================================
'''ADMM layer executes the ADMM operation on each neural network layer (e.g., linear, relu, etc)'''

class ADMM_Layer():
    def __init__(self, x_init, y_init, z_init, nn_layer, lb = None, ub = None):
        self.x = x_init
        self.y = y_init
        self.z = z_init

        self.x_init, self.y_init, self.z_init = x_init, y_init, z_init

        # save the values of y, z in the previous update
        self.y_old = y_init
        self.z_old = z_init

        self.layer = nn_layer
        self.lb = lb
        self.ub = ub

        self.num_batches = self.x.size(0)

        # define dual variables
        self.lam = torch.zeros_like(self.x)
        self.mu = torch.zeros_like(self.z)

        # variables for spectral penalty parameter selection
        self.lam_hat =  torch.zeros_like(self.x)
        self.mu_hat = torch.zeros_like(self.z)

        # over relaxation parameter
        self.alpha = 1.0

        if isinstance(self.layer, nn.Linear):
            self.label = 'linear'
        elif isinstance(self.layer, nn.ReLU):
            self.label = 'relu'
        else:
            self.label = 'undecided'

        if isinstance(self.layer, nn.Linear):
            layer = self.layer
            self.W = layer.weight
            self.b = layer.bias
            self.inv_mat = torch.inverse(torch.eye(self.W.shape[1]).to(self.W.device) + (self.W.t()).mm(self.W))

    def assign_bounds(self, lb, ub):
        self.lb = lb
        self.ub = ub

    def update_x(self, z_pre, mu_pre):
        x_old = self.x
        x_new = 1 / 2 * (self.y - self.lam + z_pre.reshape(self.y.size()) - mu_pre.reshape(self.y.size()))
        self.x = x_new

    def update_yz(self, x_next):
        self.y_old = self.y
        self.z_old = self.z

        # over-relaxation
        alpha = self.alpha
        y_before_proj = alpha*self.x + (1.0 - alpha)*self.y + self.lam
        z_before_proj = alpha*x_next.reshape(self.mu.size()) + (1.0 - alpha)*self.z + self.mu

        y_new, z_new = self.project_yz(y_before_proj, z_before_proj)

        self.y = y_new
        self.z = z_new


    def update_dual(self, x_next):
        # over relaxation
        alpha = self.alpha
        lam_new = self.lam + alpha*self.x + (1.0 - alpha)*self.y_old - self.y
        mu_new = self.mu + alpha*x_next.reshape(self.mu.size()) +(1.0 - alpha)*self.z_old - self.z

        # fixme: not sure if spectral penalty parameter selection works well with over relaxation
        lam_hat = self.lam + alpha*self.x + (1.0 - alpha)*self.y_old - self.y_old
        mu_hat =  self.mu + alpha*x_next.reshape(self.mu.size()) +(1.0 - alpha)*self.z_old - self.z_old

        self.lam = lam_new
        self.mu = mu_new

        self.lam_hat = lam_hat
        self.mu_hat

    def project_yz(self, y_before_proj, z_before_proj):
        if self.label == 'linear':
            W = self.W
            b = self.b.reshape(1, -1)
            b = b.repeat(self.y.shape[0], 1)
            temp = y_before_proj + (z_before_proj - b).matmul(W)
            y_proj = temp.matmul(self.inv_mat)
            z_proj = y_proj.matmul(W.t()) + b

        elif self.label == 'relu':
            y_proj, z_proj = relu_projection(y_before_proj, z_before_proj, self.lb, self.ub)

        else:
            raise NotImplementedError

        return y_proj, z_proj


class ADMM_Input():
    # input layer
    # fixme: How to do the projection onto a polyhedron input set?
    # z = x0 is the added concensus constraint to separate the projection operation onto the input set
    def __init__(self, z_init, z_lb, z_ub):
        self.z = z_init
        self.z_old = z_init
        self.z_init = z_init

        self.mu = torch.zeros_like(self.z)
        self.mu_hat = torch.zeros_like(self.z)

        # box input set
        self.x_lb = z_lb
        self.x_ub = z_ub

        # over relaxation parameter
        self.alpha = 1.0

    def update_yz(self, x_next):
        self.z_old = self.z

        alpha = self.alpha
        temp = alpha*x_next + (1.0 - alpha)*self.z + self.mu
        z_new = torch.max(torch.min(temp, self.x_ub), self.x_lb)
        self.z = z_new

    def update_dual(self, x_next):
        alpha = self.alpha
        mu_new = self.mu + alpha*x_next.reshape(self.mu.size()) + (1.0 - alpha)*self.z_old - self.z
        mu_hat = self.mu + alpha*x_next.reshape(self.mu.size()) + (1.0 - alpha)*self.z_old - self.z_old
        self.mu = mu_new
        self.mu_hat = mu_hat


class ADMM_Output():
    # output layer
    # fixme: the format of rho needs to be clarified
    def __init__(self, x_init, c, rho):
        # rho
        self.x = x_init
        self.x_init = x_init

        self.c = c
        self.rho = rho * torch.ones(c.size(0)).to(self.c.device)

        # over relaxation parameter
        self.alpha = 1.0

    def update_x(self, z_pre, mu_pre):
        # make sure the multiplication is done in batches
        rho = self.rho.repeat(self.c.size(1), 1).t()
        x_new = -1 / rho * self.c + z_pre - mu_pre

        self.x = x_new

    def set_objective(self, c):
        self.c = c

    def set_rho(self, rho):
        self.rho = rho * torch.ones(self.c.size(0)).to(self.c.device)

# =============================================================================
# ADMM module
# =============================================================================

class ADMM_Module():
    '''ADMM module ensembles several ADMM layers sequentially'''
    # fixme: adaptive rho has not been applied. Need to add a rho variable for each module/layer.
    def __init__(self, ADMM_layer_list, pre_act_bds_list = None):
        self.ADMM_layers = ADMM_layer_list
        self.num_layers = len(ADMM_layer_list)
        self.assign_pre_activation_bounds(pre_act_bds_list)
        self.num_batches = self.ADMM_layers[0].x.size(0)

        act_layer_count = 0
        for layer in self.ADMM_layers:
            if layer.label == 'relu':
                act_layer_count += 1

        self.num_act_layers = act_layer_count

    # define the interfaces between connected ADMM modules for ADMM updates
    @property
    def x(self):
        return self.ADMM_layers[0].x

    @x.setter
    def x(self, value):
        self.ADMM_layers[0].x = value

    @property
    def z(self):
        return self.ADMM_layers[-1].z

    @property
    def z_old(self):
        return self.ADMM_layers[-1].z_old

    @property
    def y(self):
        return self.ADMM_layers[0].y

    @property
    def y_old(self):
        return self.ADMM_layers[0].y_old

    @property
    def mu(self):
        return self.ADMM_layers[-1].mu

    @property
    def lam(self):
        return self.ADMM_layers[0].lam

    def assign_pre_activation_bounds(self, pre_act_bds_list):
        self.pre_act_bds = pre_act_bds_list
        if pre_act_bds_list is not None:
            # assing lb and ub for each ReLU layer sequentially
            count = 0
            for layer in self.ADMM_layers:
                if layer.label == 'relu':
                    pre_act_lb_and_ub = pre_act_bds_list[count]
                    layer.assign_bounds(pre_act_lb_and_ub['lb'].to(self.x.device), pre_act_lb_and_ub['ub'].to(self.x.device))
                    count += 1

            if count < len(pre_act_bds_list):
                warnings.warn('Pre-activation bounds assignment mismatch.')

    def assign_over_relaxation_parameter(self, alpha):
        for layer in self.ADMM_layers:
            layer.alpha = alpha

    def extract_pre_activation_bounds(self):
        bds_list = []
        for layer in self.ADMM_layers:
            if layer.label == 'relu':
                box = {'lb': layer.lb, 'ub': layer.ub}
                bds_list.append(box)

        return bds_list

    def update_x(self, z_pre, mu_pre):
        # for hidden module: z_pre, mu_pre are the z, mu parameters from the prvevious ADMM module
        # for input module: z_pre, mu_pre = x_lb, x_ub
        for layer in self.ADMM_layers:
            layer.update_x(z_pre, mu_pre)
            z_pre, mu_pre = layer.z, layer.mu

    def update_yz(self, x_next_module):
        # x_next_module is the x variable from the next ADMM module
        for i in range(len(self.ADMM_layers)-1):
            layer = self.ADMM_layers[i]
            x_next = self.ADMM_layers[i+1].x
            layer.update_yz(x_next)

        layer = self.ADMM_layers[-1]
        layer.update_yz(x_next_module)

    def update_dual(self, x_next_module):
        # x_next_module is the x variable from the next ADMM module
        for i in range(len(self.ADMM_layers)-1):
            layer = self.ADMM_layers[i]
            x_next = self.ADMM_layers[i+1].x
            layer.update_dual(x_next)

        layer = self.ADMM_layers[-1]
        layer.update_dual(x_next_module)

    def primal_residual(self):
        num_batches = self.num_batches
        rp_sq = 0
        for i in range(self.num_layers - 1):
            cur_layer = self.ADMM_layers[i]
            next_layer = self.ADMM_layers[i+1]
            rp_sq += torch.norm(cur_layer.y.reshape(num_batches, -1) - cur_layer.x.reshape(num_batches, -1), 2, dim=1) ** 2
            rp_sq += torch.norm(next_layer.x.reshape(num_batches, -1) - cur_layer.z.reshape(num_batches, -1), 2, dim=1) ** 2

        return rp_sq

    def dual_residual(self):
        num_batches = self.num_batches
        num_layers = self.num_layers

        rd_sq = 0
        for i in range(num_layers-1):
            cur_layer = self.ADMM_layers[i+1]
            pre_layer = self.ADMM_layers[i]

            z_diff = pre_layer.z - pre_layer.z_old
            y_diff = cur_layer.y - cur_layer.y_old
            rd_sq += torch.norm(z_diff.reshape(num_batches, -1) + y_diff.reshape(num_batches, -1), 2, dim = 1)**2

        return rd_sq


    def stopping_threshold_primal(self):
        dim_sum = 0
        x_norm_sq = 0
        yz_norm_sq = 0

        num_batches = self.num_batches
        for layer in self.ADMM_layers:
            x_norm_sq += 2*torch.norm(layer.x.reshape(num_batches, -1), 2, dim = 1)**2
            dim_sum += 2*layer.x.view(num_batches, -1).size(1)
            yz_norm_sq += torch.norm(layer.y.reshape(num_batches, -1), 2, dim = 1)**2 + torch.norm(layer.z.reshape(num_batches, -1), 2, dim = 1)**2

        return dim_sum, x_norm_sq, yz_norm_sq


    def stopping_threshold_dual(self):
        dual_norm_sq = 0
        dim_sum = 0

        num_batches = self.num_batches
        for i in range(self.num_layers-1):
            cur_layer = self.ADMM_layers[i+1]
            pre_layer = self.ADMM_layers[i]
            dual_sum = cur_layer.lam + pre_layer.mu
            dual_norm_sq += torch.norm(dual_sum.reshape(num_batches, -1), 2, dim = 1)**2
            dim_sum += dual_sum.view(num_batches, -1).size(1)*2

        return dim_sum, dual_norm_sq

    def adjust_dual_variable(self, incr_idx, incr, decr_idx, decr):
        # adjust the dual variables if residual balancing (or adaptive rho) is applied
        length = self.num_layers
        for i in range(length):
            cur_layer = self.ADMM_layers[i]
            lam = cur_layer.lam
            mu = cur_layer.mu

            lam[incr_idx] = lam[incr_idx] / incr
            lam[decr_idx] = lam[decr_idx] * decr

            mu[incr_idx] = mu[incr_idx] / incr
            mu[decr_idx] = mu[decr_idx] * decr

            cur_layer.lam = lam
            cur_layer.mu = mu

    # functions requried in the spectral penalty parameter selection
    def extract_x_var(self):
        x_var_list = []
        for layer in self.ADMM_layers:
            x_var = layer.x
            x_var_list.append(x_var)
        return x_var_list

    def extract_yz_var(self):
        # extract the (y,z) variables across the layers in a list in the order [y_0, z_0, y_1, z_1, ...]
        yz_var_list = []
        for layer in self.ADMM_layers:
            y_var = layer.y
            z_var = layer.z
            yz_var_list += [y_var, z_var]

        return yz_var_list

    def extract_dual_var(self):
        # extract the dual variables (lambda, mu) across the layers in a list in the order [lambda_0, mu_0, lambda_1, mu_1, ...]

        dual_var_list = []
        for layer in self.ADMM_layers:
            lambda_var = layer.lam
            mu_var = layer.mu
            dual_var_list += [lambda_var, mu_var]
        return dual_var_list

    def extract_dual_hat_var(self):
        # extract the hat dual variables (hat{lambda}, hat{mu}) across the layers in a list in the order [hat{lambda}_0, hat{mu}_0, hat{lambda}_1, hat{mu}_1, ...]
        # the hat dual variables are used in the spectral penalty parameter selection

        dual_var_list = []
        for layer in self.ADMM_layers:
            lambda_var = layer.lam_hat
            mu_var = layer.mu_hat
            dual_var_list += [lambda_var, mu_var]
        return dual_var_list

    def adjust_dual_variables_with_rho(self, multiplier):
        # multiplier = old_rho/new_rho
        for layer in self.ADMM_layers:
            lam = layer.lam
            mu = layer.mu
            lam_hat = layer.lam_hat
            mu_hat = layer.mu_hat

            layer.lam = lam*multiplier
            layer.mu = mu*multiplier
            layer.lam_hat = lam_hat*multiplier
            layer.mu_hat = mu_hat*multiplier


# =============================================================================
# ADMM block
# =============================================================================

class ADMM_Block():
    def __init__(self):
        pass

# =============================================================================
# ADMM block
# =============================================================================
class ADMM_Session():
    def __init__(self, ADMM_block_list, x_lb, x_ub, c, rho, alg_options = None):
        self.ADMM_blocks = ADMM_block_list
        self.num_blocks = len(ADMM_block_list)
        self.num_batches = self.ADMM_blocks[0].num_batches
        self.input_layer = ADMM_Input(self.ADMM_blocks[0].x, x_lb, x_ub)
        self.output_layer = ADMM_Output(self.ADMM_blocks[-1].z, c, rho)

        if alg_options is not None:
            self.eps_abs = alg_options['eps_abs']
            self.eps_rel = alg_options['eps_rel']
        else:
            self.eps_abs = 1e-4
            self.eps_rel = 1e-3

        self.c = c

        # primal and dual residuals: rp and rd
        self.rp = None
        self.rd = None

        # assign over relaxation parameter
        self.alpha = 1.0

    @property
    def input(self):
        return self.input_layer.z

    @property
    def output(self):
        return self.output_layer.x

    @property
    def rho(self):
        return self.output_layer.rho

    @rho.setter
    def rho(self, value):
        self.output_layer.rho = value

    def set_rho(self, value):
        self.output_layer.set_rho(value)

    def set_objective(self, c):
        self.c = c
        self.output_layer.set_objective(c)

    def assign_pre_activation_bounds(self, bounds_list):
        count = 0
        for block in self.ADMM_blocks:
            num_act_layers = block.num_act_layers
            bds = bounds_list[count:count+num_act_layers]
            block.assign_pre_activation_bounds(bds)
            count += num_act_layers

    def extract_pre_activation_bounds(self):
        pre_act_bds_list = []
        for block in self.ADMM_blocks:
            bds_list = block.extract_pre_activation_bounds()
            pre_act_bds_list += bds_list

        return pre_act_bds_list

    def assign_over_relaxation_parameter(self, alpha):
        self.alpha = alpha
        for block in self.ADMM_blocks:
            block.assign_over_relaxation_parameter(alpha)

        self.input_layer.alpha = alpha
        self.output_layer.alpha = alpha

    def update_x(self):
        z_pre, mu_pre = self.input_layer.z, self.input_layer.mu
        for i in range(self.num_blocks):
            current_block = self.ADMM_blocks[i]
            current_block.update_x(z_pre, mu_pre)
            z_pre, mu_pre = current_block.z, current_block.mu

        self.output_layer.update_x(z_pre, mu_pre)

    def update_yz(self):
        x_next = self.ADMM_blocks[0].x
        self.input_layer.update_yz(x_next)

        for i in range(self.num_blocks-1):
            x_next = self.ADMM_blocks[i+1].x
            self.ADMM_blocks[i].update_yz(x_next)

        x_next = self.output_layer.x
        self.ADMM_blocks[-1].update_yz(x_next)

    def update_dual(self):
        x_next = self.ADMM_blocks[0].x
        self.input_layer.update_dual(x_next)

        for i in range(self.num_blocks - 1):
            x_next = self.ADMM_blocks[i + 1].x
            self.ADMM_blocks[i].update_dual(x_next)

        x_next = self.output_layer.x
        self.ADMM_blocks[-1].update_dual(x_next)

    def primal_residual(self):
        rp_sq = 0
        # sum from inside of each ADMM module
        for i in range(self.num_blocks):
            block = self.ADMM_blocks[i]
            rp_sq_block = block.primal_residual()
            rp_sq += rp_sq_block

        # sum the primal residual errors from between the ADMM blocks
        num_batches = self.num_batches
        rp_sq += torch.norm(self.input_layer.z.reshape(num_batches, -1) - self.ADMM_blocks[0].x.reshape(num_batches, -1), 2, dim=1) ** 2

        for i in range(self.num_blocks-1):
            cur_block = self.ADMM_blocks[i]
            next_block = self.ADMM_blocks[i+1]
            rp_sq += torch.norm(next_block.x.reshape(num_batches, -1) - cur_block.z.reshape(num_batches, -1), 2, dim=1) ** 2

        rp_sq += torch.norm(self.ADMM_blocks[-1].z.reshape(num_batches, -1) - self.output_layer.x.reshape(num_batches, -1), 2, dim=1) ** 2
        rp = torch.sqrt(rp_sq)
        self.rp = rp
        return rp

    def dual_residual(self):
        rd_sq = 0

        # sum from inside each ADMM blocks
        for i in range(self.num_blocks):
            block = self.ADMM_blocks[i]
            rd_sq_block = block.dual_residual()
            rd_sq += rd_sq_block

        # sum from between ADMM blocks
        num_batches = self.num_batches
        z_diff = self.input_layer.z - self.input_layer.z_old
        y_diff = self.ADMM_blocks[0].y - self.ADMM_blocks[0].y_old
        rd_sq += torch.norm(z_diff.reshape(num_batches, -1) + y_diff.reshape(num_batches, -1), 2, dim=1) ** 2

        for i in range(self.num_blocks-1):
            cur_block = self.ADMM_blocks[i+1]
            pre_block = self.ADMM_blocks[i]
            z_diff = pre_block.z - pre_block.z_old
            y_diff = cur_block.y - cur_block.y_old
            rd_sq += torch.norm(z_diff.reshape(num_batches, -1) + y_diff.reshape(num_batches, -1), 2, dim=1) ** 2

        z_diff = self.ADMM_blocks[-1].z - self.ADMM_blocks[-1].z_old
        rd_sq += torch.norm(z_diff.reshape(num_batches, -1), 2, dim = 1)**2

        # fixme: check whether the multiplication of rho needs to be changed when rho is a tensor
        rd = self.rho*torch.sqrt(rd_sq)
        self.rd = rd
        return rd

    def stopping_threshold_primal(self, eps_abs, eps_rel):
        dim_sum = 0
        x_norm_sum = 0
        yz_norm_sum = 0
        for block in self.ADMM_blocks:
            dim, x_norm_sq, yz_norm_sq = block.stopping_threshold_primal()
            dim_sum += dim
            x_norm_sum += x_norm_sq
            yz_norm_sum += yz_norm_sq

        num_batches = self.num_batches
        # consider input layer
        yz_norm_sum += torch.norm(self.input_layer.z.reshape(num_batches, -1), 2, dim = 1)**2

        # consider output layer
        x_norm_sum += torch.norm(self.output_layer.x.reshape(num_batches, -1), 2, dim = 1)**2
        dim_sum += self.output_layer.x.view(num_batches, -1).size(1)

        x_norm = torch.sqrt(x_norm_sum)
        yz_norm = torch.sqrt(yz_norm_sum)
        max_norm = torch.max(x_norm, yz_norm).to(x_norm.device)
        primal_threshold = (torch.sqrt(torch.tensor([dim_sum], dtype=torch.float32))*eps_abs).to(x_norm.device) + eps_rel*max_norm
        return primal_threshold

    def stopping_threshold_dual(self, eps_abs, eps_rel):
        dim_sum = 0
        dual_norm_sum = 0
        num_batches = self.num_batches

        # consider the input layer
        cur_layer = self.ADMM_blocks[0]
        pre_layer = self.input_layer
        dual_var_sum = cur_layer.lam + pre_layer.mu
        dual_norm_sum += torch.norm(dual_var_sum.reshape(num_batches, -1), 2, dim = 1)**2
        dim_sum += dual_var_sum.view(num_batches, -1).size(1)

        # consider connections between ADMM blocks
        for i in range(self.num_blocks-1):
            cur_block = self.ADMM_blocks[i+1]
            pre_block = self.ADMM_blocks[i]
            dual_var_sum = cur_block.lam + pre_block.mu
            dual_norm_sum += torch.norm(dual_var_sum.reshape(num_batches, -1), 2, dim = 1)**2
            dim_sum += dual_var_sum.view(num_batches, -1).size(1)

        # consider the output layer
        cur_block = self.ADMM_blocks[-1]
        dual_var_sum = cur_block.mu
        dual_norm_sum += torch.norm(dual_var_sum.reshape(num_batches, -1), 2, dim=1)**2
        dim_sum += dual_var_sum.view(num_batches, -1).size(1)

        # sum the dual errors inside each ADMM block
        for i in range(self.num_blocks):
            cur_block = self.ADMM_blocks[i]
            dim, dual_norm_sq = cur_block.stopping_threshold_dual()
            dual_norm_sum += dual_norm_sq
            dim_sum += dim

        dual_threshold = (torch.sqrt(torch.tensor([dim_sum], dtype=torch.float32))*eps_abs).to(dual_norm_sum.device) + eps_rel*torch.sqrt(dual_norm_sum)
        return dual_threshold

    def residual_balancing(self, mu = 10, incr = 2, decr = 2):
        rp, rd = self.rp, self.rd

        rho = self.rho
        incr_idx = (rp > mu * rd)
        decr_idx = (rd >= mu * rp)

        if torch.any(incr_idx == True) or torch.any(decr_idx == True):
            print('')

        rho[incr_idx] = rho[incr_idx] * incr
        rho[decr_idx] = rho[decr_idx] / decr

        self.adjust_dual_variable(incr_idx, incr, decr_idx, decr)
        self.rho = rho

    def adjust_dual_variable(self, incr_idx, incr, decr_idx, decr):
        # adjust the dual variable in the input layer
        layer = self.input_layer
        mu = layer.mu
        mu[incr_idx] = mu[incr_idx] / incr
        mu[decr_idx] = mu[decr_idx] * decr
        layer.mu = mu

        # adjust the dual variables in the hidden layers
        for block in self.ADMM_blocks:
            block.adjust_dual_variable(incr_idx, incr, decr_idx, decr)


    # implement spectral penalty parameter selection
    def extract_x_var(self):
        # extract the x-variable across the ADMM layers in a list
        x_var_list = []
        for block in self.ADMM_blocks:
            x_var = block.extract_x_var()
            x_var_list += x_var

        x_var = self.output_layer.x
        x_var_list.append(x_var)
        return x_var_list

    def extract_yz_var(self):
        yz_var_list = [self.input_layer.z]

        for block in self.ADMM_blocks:
            yz_var = block.extract_yz_var()
            yz_var_list += yz_var
        return yz_var_list

    def extract_dual_var(self):
        dual_var_list = [self.input_layer.mu]
        for block in self.ADMM_blocks:
            dual_var = block.extract_dual_var()
            dual_var_list += dual_var
        return dual_var_list

    def extract_dual_hat_var(self):
        dual_var_list = [self.input_layer.mu_hat]
        for block in self.ADMM_blocks:
            dual_var = block.extract_dual_hat_var()
            dual_var_list += dual_var
        return dual_var_list

    def extract_all_var(self):
        rho = self.rho

        x_var_list = self.extract_x_var()
        yz_var_list = self.extract_yz_var()
        dual_var_list = self.extract_dual_var()
        dual_hat_var_list = self.extract_dual_hat_var()

        # recover the unscaled dual variable
        # fixme: not sure if recovering the unscaled dual variables is correct
        dual_var_list = [var*rho.view(self.num_batches,-1) for var in dual_var_list]
        dual_hat_var_list = [var*rho.view(self.num_batches,-1) for var in dual_hat_var_list]

        all_var = {'x': x_var_list, 'yz': yz_var_list, 'dual': dual_var_list, 'dual_hat': dual_hat_var_list}
        return all_var

    def adjust_dual_variables_with_rho(self, new_rho):
        # adjust the dual variables corresponding to the updated rho parameter
        old_rho = self.rho
        multiplier = old_rho/new_rho
        multiplier = multiplier.view(self.num_batches,-1)

        layer = self.input_layer
        mu = layer.mu
        mu_hat = layer.mu_hat

        layer.mu = multiplier*mu
        layer.mu_hat = multiplier*mu_hat

        # adjust the dual variables in the hidden layers
        for block in self.ADMM_blocks:
            block.adjust_dual_variables_with_rho(multiplier)

    # fixme: residual balancing through spectral penalty remains to be tested
    # def adjust_rho_with_spectral_penalty_parameters(self, var_set_new, var_set_old):
    #     old_rho = self.rho
    #
    #     diff_x = [var_set_new['x'][i] - var_set_old['x'][i] for i in range(len(var_set_new['x']))]
    #     diff_yz = [var_set_new['yz'][i] - var_set_old['yz'][i] for i in range(len(var_set_new['yz']))]
    #     diff_dual = [var_set_new['dual'][i] - var_set_old['dual'][i] for i in range(len(var_set_new['dual']))]
    #     diff_dual_hat = [var_set_new['dual_hat'][i] - var_set_old['dual_hat'][i] for i in range(len(var_set_new['dual_hat']))]
    #
    #     # computer the required inner products
    #
    #     Delta_H = torch.cat((diff_x[0], diff_x[0]), 1)
    #     for i in range(len(diff_x)-2):
    #         Delta_H = torch.cat((Delta_H, diff_x[i+1], diff_x[i+1]), 1)
    #     Delta_H = torch.cat((Delta_H, diff_x[-1]), -1)
    #
    #     Delta_dual_hat = diff_dual_hat[0]
    #     for i in range(len(diff_dual_hat)-1):
    #         Delta_dual_hat = torch.cat((Delta_dual_hat, diff_dual_hat[i+1]), 1)
    #
    #     Delta_G = -diff_yz[0]
    #     for i in range(len(diff_yz)-1):
    #         Delta_G = torch.cat((Delta_G, -diff_yz[i+1]), 1)
    #
    #     Delta_dual = diff_dual[0]
    #     for i in range(len(diff_dual)-1):
    #         Delta_dual = torch.cat((Delta_dual, diff_dual[i+1]), 1)
    #
    #     prod_dual_hat = torch.sum(torch.mul(Delta_dual_hat, Delta_dual_hat), 1)
    #     prod_H_dual_hat = torch.sum(torch.mul(Delta_H, Delta_dual_hat), 1)
    #     prod_H = torch.sum(torch.mul(Delta_H, Delta_H), 1)
    #
    #     prod_dual = torch.sum(torch.mul(Delta_dual, Delta_dual), 1)
    #     prod_G = torch.sum(torch.mul(Delta_G, Delta_G), 1)
    #     prod_G_dual = torch.sum(torch.mul(Delta_G, Delta_dual), 1)
    #
    #     # find spectral penalty parameters
    #     alpha_MG = prod_H_dual_hat/prod_H
    #     alpha_SD = prod_dual_hat/prod_H_dual_hat
    #
    #     beta_MG = prod_G_dual/prod_G
    #     beta_SD = prod_dual/prod_G_dual
    #
    #     alpha_ind = (2*alpha_MG > alpha_SD)
    #     alpha = torch.zeros(alpha_MG.size()).to(alpha_ind.device)
    #     alpha[alpha_ind] = alpha_MG[alpha_ind]
    #     alpha[~alpha_ind] = (alpha_SD - alpha_MG/2)[~alpha_ind]
    #
    #     beta_ind = (2*beta_MG > beta_SD)
    #     beta = torch.zeros(beta_MG.size()).to(alpha_ind.device)
    #     beta[beta_ind] = beta_MG[beta_ind]
    #     beta[~beta_ind] = (beta_SD - beta_MG/2)[~beta_ind]
    #
    #     # correlation
    #     alpha_cor = prod_H_dual_hat/torch.sqrt(prod_H)/torch.sqrt(prod_dual_hat)
    #     beta_cor = prod_G_dual/torch.sqrt(prod_G)/torch.sqrt(prod_dual)
    #
    #     # the default correlation threshold
    #     eps_cor = 0.2
    #
    #     ind_set_1 = (alpha_cor > eps_cor)*(beta_cor > eps_cor)
    #     ind_set_2 = (alpha_cor > eps_cor)*(beta_cor <= eps_cor)
    #     ind_set_3 = (alpha_cor <= eps_cor)*(beta_cor > eps_cor)
    #     ind_set_4 = (alpha_cor <= eps_cor)*(beta_cor <= eps_cor)
    #
    #     new_rho = torch.zeros(old_rho.size()).to(self.rho.device)
    #     new_rho[ind_set_1] = torch.sqrt(alpha*beta)[ind_set_1]
    #     new_rho[ind_set_2] = alpha[ind_set_2]
    #     new_rho[ind_set_3] = beta[ind_set_3]
    #     new_rho[ind_set_4] = old_rho[ind_set_4]
    #
    #     assert torch.all(new_rho >0)
    #
    #     self.adjust_dual_variables_with_rho(new_rho)
    #     self.rho = new_rho









# =============================================================================
# initialize the ADMM session
# =============================================================================
class InitModule():
    # auxiliary class to initialize an ADMM module from a list of NN layers
    def __init__(self, nn_layer_list, x_input, x_lb = None, x_ub = None, pre_act_bds_list = None):
        self.nn_layers = nn_layer_list
        self.input = x_input
        self.length = len(nn_layer_list)
        self.intermediate_states = []
        self.propagate()
        # interface
        self.output = self.intermediate_states[-1]

        # input lower and upper bounds
        self.input_lb = x_lb
        self.input_ub = x_ub

        # given pre activation bounds
        self.pre_act_bds = pre_act_bds_list

        # lbs and ubs saves pre-layer bounds (not only for activation layers)
        self.lbs = None
        self.ubs = None
        # find intermediate bounds for each layer through IBP if the input bounds are not None
        self.IBP()

        # interface
        if self.lbs is not None:
            self.output_lb = self.lbs[-1]
            self.output_ub = self.ubs[-1]

        # initialized ADMM module
        self.ADMM_module = None
        self.num_batches = self.input.size(0)

    def propagate(self):
        # implement a forward pass through the nn layers
        x = self.input
        output_list = [x]
        for i in range(self.length):
            layer = self.nn_layers[i]
            if isinstance(layer, nn.ReLU):
                # to avoid the in-place operation of the ReLU layer
                y = F.relu(x)
            else:
                y = layer(x)

            output_list.append(y)
            x = y

        self.intermediate_states = output_list
        return output_list

    def IBP(self):
        # find the intermediate layer bounds through interval bound propagation
        lb = self.input_lb
        ub = self.input_ub
        if (lb is not None) and (ub is not None):
            lbs, ubs = compute_bounds_interval_arithmetic(self.nn_layers, lb, ub)

        self.lbs = lbs
        self.ubs = ubs

        return lbs, ubs

    def init_ADMM_module(self):
        admm_layer_list = []
        for i in range(self.length):
            layer = self.nn_layers[i]
            x_init = self.intermediate_states[i]
            y_init = x_init
            z_init = self.intermediate_states[i+1]
            lb = self.lbs[i]
            ub = self.ubs[i]
            new_ADMM_layer = ADMM_Layer(x_init, y_init, z_init, layer, lb, ub)
            admm_layer_list.append(new_ADMM_layer)

        admm_module = ADMM_Module(admm_layer_list, self.pre_act_bds)
        return admm_module

def init_sequential_admm_session(nn_layer_list, horizon, ref_x, x_lb, x_ub, c, rho, alg_options = None):
    # fixme: currently the nn_model is considered as an instance of nn.Sequential; the preactivation bounds are not given

    x_input = ref_x
    input_lb, input_ub = x_lb.to(ref_x.device), x_ub.to(ref_x.device)

    admm_modules_list = []
    for i in range(horizon):
        init_module = InitModule(nn_layer_list, x_input, input_lb, input_ub, pre_act_bds_list = None )
        admm_module = init_module.init_ADMM_module()
        admm_modules_list.append(admm_module)

        x_input = init_module.output
        input_lb = init_module.output_lb
        input_ub = init_module.output_ub

    c = c.to(ref_x.device)
    admm_session = ADMM_Session(admm_modules_list, x_lb, x_ub, c, rho, alg_options)
    return admm_session

#
#
# # initialize a feedforward fully connected NN
# def initialize_sequential_admm_session_old(nn_model, horizon, ref_x, x_lb, x_ub, c, rho, alg_options):
#     # the pre-activation bounds have not been assigned yet
#     x = ref_x
#     admm_modules_list = []
#     for i in range(horizon):
#         admm_module, output = initialize_admm_module_from_nn(nn_model, x)
#         admm_modules_list.append(admm_module)
#         x = output
#
#     admm_session = ADMM_Session(admm_modules_list, x_lb, x_ub, c, rho, alg_options)
#     return admm_session
#
# def initialize_admm_module_from_nn(nn_model, x):
#     # nn_model is a nn.Sequential model
#     nn_layers_list = list(nn_model)
#     admm_layer_list = []
#     for i in range(len(nn_layers_list)):
#         nn_layer = nn_layers_list[i]
#         x_init = x
#         y_init = x_init
#         z_init = nn_layer(y_init)
#         admm_layer = ADMM_Layer(x_init, y_init, z_init, nn_layer)
#         admm_layer_list.append(admm_layer)
#         x = z_init
#
#     admm_module = ADMM_Module(admm_layer_list)
#     output = x
#     return admm_module, output


# =============================================================================
# Custom layers
# =============================================================================

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# =============================================================================
# projection functions to the convex overapproximation of each layer
# =============================================================================

def relu_projection(y_0, z_0, y_min, y_max):
    # flatten the input to the ReLU layer
    sz = y_0.size()
    yp, zp = relu_proj_convex_hull(y_0.reshape(-1), z_0.reshape(-1), y_min.reshape(-1), y_max.reshape(-1))
    return yp.view(*sz), zp.view(*sz)


def relu_proj_convex_hull(y_before_proj, z_before_proj, y_lb, y_ub):
    # projected vectors
    y_proj = torch.zeros_like(y_before_proj)
    z_proj = torch.zeros_like(z_before_proj)

    z_lb = torch.max(y_lb, torch.zeros_like(y_lb))
    z_ub = torch.max(y_ub, torch.zeros_like(y_ub))

    # case 1: y_lb >= 0
    indices = (y_lb >= 0)

    mid = (y_before_proj[indices] + z_before_proj[indices]) / 2
    mid = torch.min(torch.max(mid, y_lb[indices]), y_ub[indices])

    y_proj[indices] = mid
    z_proj[indices] = mid

    # case 2: y_max <= 0
    indices = (y_ub <= 0)
    mid = torch.min(torch.max(y_before_proj[indices], y_lb[indices]), y_ub[indices])
    mid = torch.min(torch.max(mid, y_lb[indices]), y_ub[indices])

    y_proj[indices] = mid
    z_proj[indices] = 0

    # case 3: y_lb <= 0 <= y_ub
    indices = (y_lb < 0) * (y_ub > 0)
    if torch.any(indices):
        y0 = y_before_proj[indices]
        z0 = z_before_proj[indices]

        y_min = y_lb[indices]
        y_max = y_ub[indices]
        z_min = z_lb[indices]
        z_max = z_ub[indices]

        N = y0.size(0)

        yp = y0.new_zeros(N, 3)
        zp = y0.new_zeros(N, 3)

        dist = y0.new_zeros(N, 3)

        # project onto z = y, 0 <= y <= y_max
        yp[:, 0] = torch.min(torch.max(0.5 * (y0 + z0), torch.zeros_like(y0)), y_max)
        zp[:, 0] = yp[:, 0]

        dist[:, 0] = 0.5 * (yp[:, 0] - y0) ** 2 + 0.5 * (zp[:, 0] - z0) ** 2

        # project onto z = 0, y_min <= y <= 0
        yp[:, 1] = torch.min(torch.max(y0, y_min), torch.zeros_like(y0))
        zp[:, 1] = 0.0

        dist[:, 1] = 0.5 * (yp[:, 1] - y0) ** 2 + 0.5 * (zp[:, 1] - z0) ** 2

        # project onto z = s*y + b, y_min <= y <= y_max
        s = (z_max - z_min) / (y_max - y_min)

        yp[:, 2] = torch.min(torch.max((y0 + s ** 2 * y_min - s * (z_min - z0)) / (s ** 2 + 1), y_min), y_max)
        zp[:, 2] = z_min + s * (yp[:, 2] - y_min)

        dist[:, 2] = 0.5 * (yp[:, 2] - y0) ** 2 + 0.5 * (zp[:, 2] - z0) ** 2

        index = torch.argmin(dist, dim=1)

        yp = yp[torch.arange(0, N), index]
        zp = zp[torch.arange(0, N), index]

        index = (z0 >= y0) * (z0 >= 0) * (z0 <= z_min + s * (y0 - y_min))

        yp[index] = y0[index]
        zp[index] = z0[index]

        # plug in the values
        y_proj[indices] = yp
        z_proj[indices] = zp

    return y_proj, z_proj


# =============================================================================
# run ADMM
# =============================================================================

def run_ADMM(ADMM_sess, alg_options = {}):
    eps_abs = alg_options.get('eps_abs', 1e-4)
    eps_rel = alg_options.get('eps_rel', 1e-3)
    # residual_balancing = alg_options.get('residual_balancing', False)
    adaptive_rho = alg_options.get('adaptive_rho', False)
    rho_update_freq = alg_options.get('rho_update_freq', 2)
    max_iter = alg_options.get('max_iter', 8000)
    record = alg_options.get('record', False)
    verbose = alg_options.get('verbose', False)
    view_id = alg_options.get('view_id', 0)
    alpha = alg_options.get('alpha', 1.0)

    start_time = time.time()
    rp_list = []
    rd_list = []
    obj_list = []
    p_tol_list = []
    d_tol_list = []
    rho_list = []
    succ_ratio_list = []

    num = ADMM_sess.num_batches
    c = ADMM_sess.c

    # assign over relaxation parameters
    ADMM_sess.assign_over_relaxation_parameter(alpha)

    buffer_var = None
    for i in range(max_iter):
        last_state = ADMM_sess.output
        product = last_state.reshape(num, -1)*c.reshape(num, -1)
        obj = product.sum(dim=1)

        ADMM_sess.update_x()
        ADMM_sess.update_yz()
        ADMM_sess.update_dual()
        rp = ADMM_sess.primal_residual()
        rd = ADMM_sess.dual_residual()

        # if residual_balancing:
        # # fixme: residual balancing hasn't been fixed yet; Do we need to stop residual balancing after a number of iterations?
        #     ADMM_sess.residual_balancing()

        if adaptive_rho:
            # if i >100:
            #     if i % rho_update_freq == 0:
            #         cur_var = ADMM_sess.extract_all_var()
            #         if buffer_var is None:
            #             buffer_var = cur_var
            #         else:
            #             ADMM_sess.adjust_rho_with_spectral_penalty_parameters(cur_var, buffer_var)
            #             buffer_var = cur_var

            if i >100:
                cur_var = ADMM_sess.extract_all_var()

                if i % rho_update_freq == 0:
                    if buffer_var is None:
                        buffer_var = cur_var
                    else:
                        ADMM_sess.adjust_rho_with_spectral_penalty_parameters(cur_var, buffer_var)

                buffer_var = cur_var


        p_tol = ADMM_sess.stopping_threshold_primal(eps_abs, eps_rel)
        d_tol = ADMM_sess.stopping_threshold_dual(eps_abs, eps_rel)
        rho = ADMM_sess.rho

        if verbose:
            # print('iter no. ', str(i), 'rho: ', ADMM_sess.rho[view_id] ,'primal: ', rp[view_id], p_tol[view_id], ' dual: ', rd[view_id], d_tol[view_id], ' obj: ', obj[view_id])
            # print('iter no. ', str(i) ,'primal (res. and tol.): ', rp[view_id].item(), p_tol[view_id].item(), ' dual (res. and tol.): ', rd[view_id].item(), d_tol[view_id].item(), ' obj: ', obj[view_id].item())
            print('iter no. {:d}, rho {:f}, primal res. {:f}, tol. {:f}, dual res. {:f}, tol {:f}, obj.{:f}'.format(i, rho[view_id], rp[view_id].item(), p_tol[view_id].item(), rd[view_id].item(), d_tol[view_id].item(), obj[view_id].item()))

        if record:
            rp_list.append(rp)
            rd_list.append(rd)
            obj_list.append(obj)
            p_tol_list.append(p_tol)
            d_tol_list.append(d_tol)
            rho_list.append(ADMM_sess.rho.clone())

        status = 'max_iter_reached'

        if torch.all(rp <= p_tol) and torch.all(rd <= d_tol):
            status = 'meet_tolerance'
            break

        if i % 100 == 0:
            succ_ratio = ((rp <= p_tol) & (rd <= d_tol)).sum().item() / num
            succ_ratio_list.append(succ_ratio)
            print('success ratio', succ_ratio)

    if status == 'meet_tolerance':
        print('All examples meet stopping criterion.')
    else:
        print('Maximum number of iteration is reached.')

    termination_example_id = ((rp <= p_tol) & (rd <= d_tol))

    running_time = time.time() - start_time

    if record:
        result = {'obj_list': obj_list, 'rp_list': rp_list, 'rd_list': rd_list, 'p_tol_list': p_tol_list, 'd_tol_list': d_tol_list,
                  'rho_list': rho_list, 'running_time': running_time,  'status': status}
    else:
        result = {'obj_list': obj, 'rp_list': rp, 'rd_list': rd, 'p_tol_list': p_tol, 'd_tol_list': d_tol, 'rho_list': rho,
                  'running_time': running_time, 'status': status}

    return obj, running_time, result, termination_example_id



def adaptive_rho(ADMM_section, rp, rd, mu=10, incr=2, decr=2):
    rho = ADMM_section[-1].rho
    incr_idx = (rp > mu * rd)
    decr_idx = (rd >= mu * rp)
    rho[incr_idx] = rho[incr_idx] * incr
    rho[decr_idx] = rho[decr_idx] / decr

    length = len(ADMM_section)
    for i in range(length - 1):
        module = ADMM_section[i]
        lam = module.lam
        mu = module.mu

        lam[incr_idx] = lam[incr_idx] / incr
        lam[decr_idx] = lam[decr_idx] * decr
        module.lam = lam

        mu[incr_idx] = mu[incr_idx] / incr
        mu[decr_idx] = mu[decr_idx] * decr
        module.mu = mu

    output_module = ADMM_section[-1]
    output_module.rho = rho



# =============================================================================
# functions for computing the intermediate bounds for NN layers
# =============================================================================

def compute_bounds_interval_arithmetic(nn_layer_list, lb0, ub0):
    # Calculate upper and lower bounds with IBP (loose calculation)
    # lb0, ub0 are the lower and upper bounds on the input
    # only box over-approximation is considered in IBP

    lbs, ubs = [lb0], [ub0]
    lbs_pre_act = []
    ubs_pre_act = []
    for layer in nn_layer_list:
        lb, ub = lbs[-1], ubs[-1]
        if isinstance(layer, nn.Linear):
            mu = (ub + lb) / 2
            r = (ub - lb) / 2

            mu = F.linear(mu, layer.weight, layer.bias)
            r = F.linear(r, layer.weight.abs())

            lbs.append(mu - r)
            ubs.append(mu + r)

        elif isinstance(layer, nn.Conv2d):
            mu = (ub + lb) / 2
            r = (ub - lb) / 2

            mu = layer(mu)

            if layer.bias is None:
                original_weight = layer.weight
                layer.weight = nn.Parameter(layer.weight.abs())
                r = layer(r)
                layer.weight = original_weight
            else:
                original_weight = layer.weight
                original_bias = layer.bias

                layer.weight = nn.Parameter(layer.weight.abs())
                layer.bias = nn.Parameter(torch.zeros(layer.bias.size()).to(r.device))
                r = layer(r)
                layer.weight = original_weight
                layer.bias = original_bias

            lbs.append(mu - r)
            ubs.append(mu + r)

        elif isinstance(layer, nn.ReLU):
            # extract the preactivation layer
            lbs_pre_act.append(lb)
            ubs_pre_act.append(ub)

            # do not use layer(lb), since the given relu layer may have inplace = True.
            lbs.append(F.relu(lb))
            ubs.append(F.relu(ub))

        elif isinstance(layer, nn.BatchNorm2d):
            mu = (ub + lb) / 2
            r = (ub - lb) / 2

            alpha = 1.0 / torch.sqrt(layer.running_var + layer.eps) * layer.weight
            beta = -layer.running_mean / torch.sqrt(layer.running_var + layer.eps) * layer.weight + layer.bias

            N, C, H, W = mu.size()

            weight = alpha.reshape(C, 1, 1)
            weight = weight.repeat(N, 1,1,1)
            bias = beta.reshape(C,1,1)
            bias = bias.repeat(N,1,1,1)

            mu = weight*mu + bias
            r = weight.abs()*r

            lbs.append(mu - r)
            ubs.append(mu + r)

        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            assert layer.output_size[0] == 1
            lbs.append(layer(lb))
            ubs.append(layer(ub))

        elif isinstance(layer, Flatten):
            lbs.append(layer(lb))
            ubs.append(layer(ub))

        else:
            raise ValueError("Unknown layer type for IBP")

    return lbs, ubs


def intermediate_bounds_from_ADMM(nn_layers_list, x0_lb, x0_ub, alg_options, file_name = None):
    # fixme: the number of batch is currently set to 1
    assert x0_lb.size(0) == 1
    if 'rho' in alg_options.keys():
        rho = alg_options['rho']
    else:
        rho = 1.0

    x0 = (x0_lb + x0_ub)/2

    # find the index of activation layers
    act_layer_ind_set =[]
    for i in range(len(nn_layers_list)):
        if isinstance(nn_layers_list[i], nn.ReLU):
            act_layer_ind_set.append(i)

    N = len(act_layer_ind_set)
    pre_act_bds_list = []
    runtime = []
    for i in tqdm(range(N), desc = 'pre-act-admm'):
        # find the input dimension to each activation layer
        nn_layers_truncation = nn_layers_list[:act_layer_ind_set[i]]
        temp_net = nn.Sequential(*nn_layers_truncation)
        output = temp_net(x0)
        output_dim = output.view(output.size(0), -1).size(1)
        x_input_batch = x0.repeat(2*output_dim, 1)
        x0_lb_batch = x0_lb.repeat(2*output_dim, 1)
        x0_ub_batch = x0_ub.repeat(2*output_dim, 1)

        init_module = InitModule(nn_layers_truncation, x_input_batch, x0_lb_batch, x0_ub_batch, pre_act_bds_list=None)
        admm_module = init_module.init_ADMM_module()
        c = torch.cat((torch.eye(output_dim), -torch.eye(output_dim))).to(x0.device)

        admm_sess = ADMM_Session([admm_module], x0_lb_batch, x0_ub_batch, c, rho)

        if len(pre_act_bds_list) >= 1:
            pre_act_bds = [{'lb': item['lb'].repeat(2*output_dim,1).to(x0.device), 'ub': item['ub'].repeat(2*output_dim,1).to(x0.device)} for item in pre_act_bds_list]
            admm_sess.assign_pre_activation_bounds(pre_act_bds)

        objective, running_time, result, termination_example_id = run_ADMM(admm_sess, alg_options)

        runtime.append(running_time)
        lb = objective[:output_dim].unsqueeze(0)
        ub = -objective[output_dim:].unsqueeze(0)

        assert torch.all(lb <= ub)
        pre_act_bds_list.append({'lb': lb.to(torch.device('cpu')), 'ub': ub.to(torch.device('cpu'))})

        if file_name is not None:
            result = {'pre_act_bds': pre_act_bds_list, 'runtime': runtime, 'alg_options': alg_options}
            torch.save(result, file_name)

    return pre_act_bds_list, runtime


def init_sequential_admm_session(nn_layer_list, horizon, ref_x, x_lb, x_ub, c, rho, alg_options = None):
    # fixme: currently the nn_model is considered as an instance of nn.Sequential; the preactivation bounds are not given

    x_input = ref_x
    input_lb, input_ub = x_lb.to(ref_x.device), x_ub.to(ref_x.device)

    admm_modules_list = []
    for i in range(horizon):
        init_module = InitModule(nn_layer_list, x_input, input_lb, input_ub, pre_act_bds_list = None )
        admm_module = init_module.init_ADMM_module()
        admm_modules_list.append(admm_module)

        x_input = init_module.output
        input_lb = init_module.output_lb
        input_ub = init_module.output_ub

    c = c.to(ref_x.device)
    admm_session = ADMM_Session(admm_modules_list, x_lb, x_ub, c, rho, alg_options)
    return admm_session

