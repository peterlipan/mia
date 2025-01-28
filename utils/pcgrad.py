import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DynamicTemperatureScheduler:
    def __init__(self, 
                 initial_temp=1.0, 
                 min_temp=0.01, 
                 max_temp=10.0,
                 decay_type='exponential',
                 decay_rate=0.999):
        """
        Dynamic temperature scheduler for gradient annealing
        
        Args:
            initial_temp (float): Starting temperature
            min_temp (float): Minimum allowable temperature
            max_temp (float): Maximum allowable temperature
            decay_type (str): Temperature decay strategy
        """
        self.temperature = initial_temp
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        
        # Tracking optimization dynamics
        self.conflict_history = []
        self.acceptance_history = []
        self.iteration_count = 0
    
    @property
    def mean_conflict(self):
        return np.mean(self.conflict_history[-10:])

    @property
    def mean_acceptance(self):
        return np.mean(self.acceptance_history[-10:])

    def update(self, conflict_intensity, acceptance_prob):
        """
        Dynamically update temperature based on optimization dynamics
        
        Args:
            conflict_intensity (float): Measure of gradient conflict
            acceptance_prob (float): Probability of gradient acceptance
        """
        self.iteration_count += 1
        self.conflict_history.append(conflict_intensity)
        self.acceptance_history.append(acceptance_prob)

        if self.decay_type == 'exponential':
            # exponentially decay temperature only based on the number of iterations
            self.temperature *= self.decay_rate
        
        elif self.decay_type == 'adaptive':
            # update temperature based on the mean and variance of the conflict history
            conflict_variance = np.var(self.conflict_history[-10:])
            acceptance_mean = np.mean(self.acceptance_history[-10:])
            
            self.temperature *= (1 - conflict_variance * (1 - acceptance_mean))
        
        elif self.decay_type == 'cyclic':
            # Oscillating temperature to maintain exploration
            self.temperature = (
                self.min_temp + 
                (self.max_temp - self.min_temp) * 
                np.sin(2 * np.pi * self.iteration_count / 100)
            )

        # Ensure temperature stays within bounds
        self.temperature = np.clip(
            self.temperature, 
            self.min_temp, 
            self.max_temp
        )

    def get_temperature(self):
        return self.temperature


class PCGrad:
    def __init__(self, optimizer, temperature=1.0, decay_rate=0.999, reduction='mean'):
        self._optim = optimizer
        self.init_temp = temperature
        self._reduction = reduction
        self._temp_scheduler = DynamicTemperatureScheduler(initial_temp=temperature, decay_rate=decay_rate)
    
    @property
    def cur_temp(self):
        return self._temp_scheduler.get_temperature()
    
    @property
    def conflict_intensity(self):
        return self._temp_scheduler.mean_conflict
    
    @property
    def acceptance_prob(self):
        return self._temp_scheduler.mean_acceptance

    @property
    def param_groups(self):
        return self._optim.param_groups

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        Clear the gradient of the parameters.
        '''
        self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        Update the parameters with the gradient.
        '''
        self._optim.step()

    def pc_backward(self, main_obj, aux_objs):
        '''
        Calculate the gradient of the parameters using PCGrad.

        input:
        - main_obj: the main objective (primary loss).
        - aux_objs: list of auxiliary objectives (secondary losses).
        '''
        if not isinstance(aux_objs, list) or len(aux_objs) == 0:
            raise ValueError("`aux_objs` must be a non-empty list of auxiliary objectives.")

        main_grad, main_shape, main_has_grad = self._fetch_main_grad(main_obj)
        aux_grads, aux_shapes, aux_has_grads = self._pack_grad(aux_objs)

        combined_grad = self._gradient_annealing(main_grad, main_has_grad, aux_grads, aux_has_grads)

        self._set_grad(combined_grad, main_shape)

    def _fetch_main_grad(self, main_obj):
        '''
        Fetch the gradient of the parameters of the network with the main optimization objective.

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of masks representing whether the parameter has a gradient
        '''
        self._optim.zero_grad(set_to_none=True)
        main_obj.backward(retain_graph=True)
        grad, shape, has_grad = self._retrieve_grad()
        return self._flatten_grad(grad, shape), shape, self._flatten_grad(has_grad, shape)

    def _gradient_annealing(self, main_grad, main_has_grad, aux_grads, aux_has_grads):
        combined_grad = main_grad.clone()
        
        aux_conflict_mean = 0
        acceptance_prob_mean = 0

        for aux_grad, aux_has_grad in zip(aux_grads, aux_has_grads):
            dot_product = torch.dot(main_grad, aux_grad)
            cos_sim = dot_product / (main_grad.norm() * aux_grad.norm() + 1e-8)
            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
            current_temp = self._temp_scheduler.get_temperature()    
            acceptance_prob = torch.exp(-cos_sim / current_temp)
            aux_conflict_mean += cos_sim.item()
            acceptance_prob_mean += acceptance_prob.item()
            
            if cos_sim >= 0:
                # scale
                aux_grad *= acceptance_prob
                
                
            else:
                               
                # Apply gradient annealing if conditions are met
                if torch.rand(1).item() > acceptance_prob:
                    aux_grad -= (dot_product / (main_grad.norm()**2 + 1e-8)) * main_grad

            # Accumulate gradients
            combined_grad[aux_has_grad.bool()] += aux_grad[aux_has_grad.bool()]
        
        # Update temperature scheduler with average conflict intensity and acceptance probability
        self._temp_scheduler.update(aux_conflict_mean / len(aux_grads), acceptance_prob_mean / len(aux_grads))

        return combined_grad



    def _set_grad(self, grads, shapes):
        '''
        Set the modified gradients to the network.
        '''
        grads = self._unflatten_grad(grads, shapes)
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1

    def _pack_grad(self, objectives):
        '''
        Pack the gradient of the parameters of the network for each objective.

        output:
        - grads: a list of the gradient of the parameters
        - shapes: a list of the shape of the parameters
        - has_grads: a list of masks representing whether the parameter has a gradient
        '''
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _retrieve_grad(self):
        '''
        Get the gradient of the parameters of the network for a specific objective.

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of masks representing whether the parameter has a gradient
        '''
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                shape.append(p.shape)
                if p.grad is None:
                    grad.append(torch.zeros_like(p))
                    has_grad.append(torch.zeros_like(p, dtype=torch.bool))
                else:
                    grad.append(p.grad.clone())
                    has_grad.append(torch.ones_like(p, dtype=torch.bool))
        return grad, shape, has_grad

    def _flatten_grad(self, grads, shapes):
        '''
        Flatten the gradients for easier manipulation.
        '''
        return torch.cat([g.flatten() for g in grads])

    def _unflatten_grad(self, grads, shapes):
        '''
        Unflatten the gradients back to their original shapes.
        '''
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self.norm = nn.LayerNorm(2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self.norm(self._linear(x))
        return self._head1(feat), self._head2(feat)
    

def main(rank, world_size, x, y):
    from datetime import timedelta
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timedelta(hours=12))
    torch.cuda.set_device(rank)

    x, y = x[rank].cuda(), y[rank].cuda()

    net = MultiHeadTestNet().cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[rank])
    net._set_static_graph()

    y_pred1, y_pred2 = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    # pc_adam.zero_grad()
    
    loss1_fn, loss2_fn = nn.MSELoss(), nn.L1Loss()
    loss1 = loss1_fn(y_pred1, y)
    loss2 = loss2_fn(y_pred2, y)

    pc_adam.pc_backward(main_obj=loss1, aux_objs=[loss2])
    for name, p in net.named_parameters():
        print(name, p.grad)
    
    # Test multi-gpu parameters updating
    if dist.is_available() and dist.is_initialized():
        for p in net.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)  # Sum gradients
                p.grad.data /= dist.get_world_size() 
    
    dist.barrier()
    if rank == 0:
        print('-' * 50, 'After all-reduce', '-' * 50)
    for name, p in net.named_parameters():
        print(name, p.grad)


if __name__ == '__main__':
    import os
    import torch.multiprocessing as mp
    world_size = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "3,4"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6666'

    torch.manual_seed(4)
    x, y = torch.randn(world_size, 2, 3), torch.randn(world_size, 2, 4)
   
    mp.spawn(main, args=(world_size,x,y,), nprocs=world_size, join=True)
