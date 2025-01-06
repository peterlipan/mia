import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DynamicTemperatureScheduler:
    def __init__(self, 
                 initial_temp=1.0, 
                 min_temp=0.01, 
                 max_temp=10.0,
                 decay_type='exponential'):
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
        
        # Tracking optimization dynamics
        self.conflict_history = []
        self.acceptance_history = []
        self.iteration_count = 0

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

        # Dynamic temperature adjustment strategies
        if self.decay_type == 'exponential':
            decay_factor = np.exp(-conflict_intensity * (1 - acceptance_prob))
            self.temperature *= decay_factor
        
        elif self.decay_type == 'adaptive':
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
    def __init__(self, optimizer, temperature=0.07, reduction='mean'):
        self._optim = optimizer
        self.init_temp = temperature
        self._reduction = reduction
        self._temp_scheduler = DynamicTemperatureScheduler(initial_temp=temperature)

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

        for aux_grad, aux_has_grad in zip(aux_grads, aux_has_grads):
            dot_product = torch.dot(main_grad, aux_grad)
            
            if dot_product != 0:
                conflict_intensity = -dot_product / (main_grad.norm() * aux_grad.norm() + 1e-8)
                current_temp = self._temp_scheduler.get_temperature()
                
                acceptance_prob = torch.exp(-conflict_intensity / current_temp)
                
                # Update temperature scheduler
                self._temp_scheduler.update(
                    conflict_intensity.item(), 
                    acceptance_prob.item()
                )
                
                if torch.rand(1).item() < acceptance_prob:
                    # With (1 - acceptance_prob) probability,USE the orthogonalized gradient
                    aux_grad = aux_grad - (dot_product / (main_grad.norm()**2 + 1e-8)) * main_grad

            # Gradient accumulation
            combined_grad[aux_has_grad.bool()] += aux_grad[aux_has_grad.bool()]

        return combined_grad

    def _set_grad(self, grads, shapes):
        '''
        Set the modified gradients to the network.
        '''
        grads = self._unflatten_grad(grads, shapes)
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is not None:  # Avoid overwriting None gradients
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

    net = TestNet().cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[rank])

    y_pred = net(x)
    pc_adam = PCGrad(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward(main_obj=loss1, aux_objs=[loss2])
    for p in net.parameters():
        print(p.grad)
    
    # Test multi-gpu parameters updating
    if dist.is_available() and dist.is_initialized():
        for p in net.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)  # Sum gradients
                p.grad.data /= dist.get_world_size() 
    
    dist.barrier()
    if rank == 0:
        print('-' * 50, 'After all-reduce', '-' * 50)
    for p in net.parameters():
        print(p.grad)


if __name__ == '__main__':
    import os
    import torch.multiprocessing as mp
    world_size = 2
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6666'

    torch.manual_seed(4)
    x, y = torch.randn(world_size, 2, 3), torch.randn(world_size, 2, 4)
   
    mp.spawn(main, args=(world_size,x,y,), nprocs=world_size, join=True)
