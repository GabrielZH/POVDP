from typing import Union
import copy
import os


import numpy as np
import blobfile
import torch

from povdp.policy.edm_policy import ConditionalEDMPolicy
from povdp.dataset.normalization import BitNormalizer
from povdp.utils.arrays import batch_to_device
from povdp.utils.timer import Timer

from povdp.utils.resample import LossAwareSampler, UniformSampler
from povdp.utils.fp16_util import *

from calvin.core.domains.gridworld.actions import GridDirs


action_list = GridDirs.DIRS_8 + [(0, 0)]

# For ImageNet experiments, this was a good default value.
# We found that the log_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


def cycle(dl):
    while True:
        for data in dl:
            yield data


def bits2int(x, n_bits):
    x = (x > 0).astype(np.int32)
    x = np.sum(x * (2 ** np.arange(n_bits)), axis=-1)
    return x


# def get_blob_logdir():
#     # You can change this to be a separate path to save checkpoints to
#     # a blobstore or some external drive.
#     return logger.get_dir()


def find_ema_checkpoint(main_checkpoint, epoch, ema_decay):
    if main_checkpoint is None:
        return None
    filename = f"policy_ema_{ema_decay}_state_{(epoch):04d}.pt"
    path = blobfile.join(
        blobfile.dirname(main_checkpoint), filename)
    if blobfile.exists(path):
        return path
    return None


def parse_resume_epoch_from_filename(filename):
    """
    Parse filenames of the form path/to/policy_ema_EMADECAY_state_NNNN.pt, where NNNN is the
    checkpoint's number of epochs.
    """
    split = filename.split("_")
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


# def log_loss_dict(diffusion, ts, losses):
#     for key, values in losses.items():
#         logger.logkv_mean(key, values.mean().item())
#         # Log the quantiles (four quartiles, in particular).
#         for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
#             quartile = int(4 * sub_t / diffusion.n_timesteps)
#             logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


class EDMTrainer(object):
    def __init__(
            self,
            edm_diffusion: ConditionalEDMPolicy, 
            train_dataset, 
            val_dataset, 
            schedule_sampler, 
            ema_decay=.995, 
            train_batch_size=32, 
            val_batch_size=1, 
            train_lr=None, 
            lr_anneal_steps=0, 
            num_train_steps=1e6, 
            log_freq=100, 
            sample_freq=1000, 
            save_freq=1000, 
            label_freq=100000, 
            use_fp16=False, 
            fp16_scale_growth=1e-3,
            results_folder='./results', 
            resume_diffusion_checkpoint=None, 
            resume_point_cloud_checkpoint=None, 
            resume_flat_map_checkpoint=None, 
            load_from_ckpt=False
    ):
        super().__init__()

        self.edm_diffusion = edm_diffusion

        self.train_dataset = train_dataset
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=train_batch_size, 
            num_workers=0, 
            shuffle=True, 
            collate_fn=self.train_dataset.collate_fn, 
            pin_memory=True
        )
        self.val_dataset = val_dataset
        self.val_dataloader = torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=val_batch_size, 
            num_workers=0, 
            shuffle=False, 
            collate_fn=self.val_dataset.collate_fn, 
            pin_memory=True
        )

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_train_steps = num_train_steps
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq

        self.ema_decay = (
            [ema_decay] if isinstance(ema_decay, float)
            else [float(x) for x in ema_decay.split(',')]
        )
        self.schedule_sampler = schedule_sampler or \
            UniformSampler(self.edm_diffusion)
        self.train_lr = train_lr
        self.lr_anneal_steps = lr_anneal_steps

        self.resume_diffusion_checkpoint = resume_diffusion_checkpoint
        self.resume_point_cloud_checkpoint = resume_point_cloud_checkpoint
        self.resume_flat_map_checkpoint = resume_flat_map_checkpoint
        if load_from_ckpt:
            if resume_diffusion_checkpoint is not None:
                self.resume_diffusion_checkpoint = os.path.join(
                    results_folder, 
                    f'policy_ema_{ema_decay}_state_{resume_diffusion_checkpoint}.pt'
                )
            if resume_point_cloud_checkpoint is not None:
                self.resume_point_cloud_checkpoint = os.path.join(
                    results_folder, 
                    f'point_cloud_encoder_{resume_point_cloud_checkpoint}.pt'
                )
                self.edm_diffusion.point_cloud_to_2d_projector.point_conv_net.load_state_dict(
                    torch.load(self.resume_point_cloud_checkpoint)
                )
                for param in self.edm_diffusion.point_cloud_to_2d_projector.point_conv_net.parameters():
                    param.requires_grad = False
            if resume_flat_map_checkpoint is not None:
                self.resume_flat_map_checkpoint = os.path.join(
                    results_folder, 
                    f'flat_map_encoder_{resume_flat_map_checkpoint}.pt'
                )
                self.edm_diffusion.point_cloud_to_2d_projector.map_conv_net.load_state_dict(
                    torch.load(self.resume_flat_map_checkpoint)
                )
                for param in self.edm_diffusion.point_cloud_to_2d_projector.map_conv_net.parameters():
                    param.requires_grad = False
            
        self.logdir = results_folder

        self.step = 0
        if self.resume_diffusion_checkpoint:
            self.resume_epoch = parse_resume_epoch_from_filename(
                self.resume_diffusion_checkpoint
            )
        else:
            self.resume_epoch = 0

        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.model_params = list(
            self.edm_diffusion.diffusion_model.parameters())
        self.master_params = self.model_params
        self.point_cloud_to_2d_params = list(
            self.edm_diffusion.point_cloud_to_2d_projector.parameters()
        ) if self.edm_diffusion.point_cloud_to_2d_projector is not None else None
        self.param_groups_and_shapes = None
        self.log_loss_scale = INITIAL_LOG_LOSS_SCALE

        if self.use_fp16:
            self.param_groups_and_shapes = get_param_groups_and_shapes(
                self.edm_diffusion.diffusion_model.named_parameters()
            )
            self.master_params = make_master_params(
                self.param_groups_and_shapes
            )
            self.edm_diffusion.diffusion_model.convert_to_fp16()

        if self.point_cloud_to_2d_params is not None \
            and self.resume_point_cloud_checkpoint is None \
                and self.resume_flat_map_checkpoint is None:
            self.optimizer = torch.optim.RAdam(
                self.master_params + self.point_cloud_to_2d_params,
                lr=self.train_lr, 
            )
        else:
            self.optimizer = torch.optim.RAdam(
                self.master_params, 
                lr=self.train_lr
            )
        
        if self.resume_epoch:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_params(decay) 
                for decay in self.ema_decay
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.master_params)
                for _ in range(len(self.ema_decay))
            ]

        self.step = self.resume_epoch * (len(self.train_dataset) // self.train_batch_size)

    def backward(self, loss: torch.Tensor):
        # for name, param in self.edm_diffusion.named_parameters():
        #     print(f"{name}")
        #     if param.grad is not None and torch.isnan(param.grad).any():
        #         print(f"NaN gradient in {name}")
        if self.use_fp16:
            loss_scale = 2 ** self.log_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

    def optimize(self):
        if self.use_fp16:
            return self._fp16_optimize()
        else:
            return self._normal_optimize()

    def _fp16_optimize(self):
        model_grads_to_master_grads(
            self.param_groups_and_shapes, 
            self.master_params)
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.log_loss_scale)
        if check_overflow(grad_norm):
            self.log_loss_scale -= 1
            print(
                f"Found NaN, decreased log_loss_scale to {self.log_loss_scale}")
            zero_master_grads(self.master_params)
            return False

        for p in self.master_params:
            p.grad.mul_(1. / (2**self.log_loss_scale))
        self.optimizer.step()
        zero_master_grads(self.master_params)
        master_params_to_model_params(self.param_groups_and_shapes, self.master_params)
        self.log_loss_scale += self.fp16_scale_growth
        
        return True
    
    def _normal_optimize(self):
        # grad_norm, param_norm = self._compute_norms()
        self.optimizer.step()
        return True
    
    def _compute_norms(self, grad_scale=1.):
        grad_norm = 0.
        param_norm = 0.
        for param in self.master_params:
            with torch.no_grad():
                param_norm += torch.norm(
                    param, 
                    p=2, 
                    dtype=torch.float32).item() ** 2
                if param.grad is not None:
                    grad_norm += torch.norm(
                        param.grad, 
                        p=2, 
                        dtype=torch.float32).item() ** 2
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)
    
    def _zero_grad(self):
        if self.point_cloud_to_2d_params is not None \
            and self.resume_point_cloud_checkpoint is None \
                and self.resume_flat_map_checkpoint is None:
            model_params = self.model_params + self.point_cloud_to_2d_params
        else:
            model_params = self.model_params
        for param in model_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    
    def master_params_to_state_dict(self, master_params):
        return master_params_to_state_dict(
            self.edm_diffusion.diffusion_model, 
            self.param_groups_and_shapes, 
            master_params, self.use_fp16
        )

    def state_dict_to_master_params(self, state_dict):
        return state_dict_to_master_params(
            self.edm_diffusion.diffusion_model, 
            state_dict, 
            self.use_fp16
        )
    
    def update_ema(self):
        for decay_rate, params in zip(self.ema_decay, self.ema_params):
            self._update_ema(
                target_params=params, 
                source_params=self.master_params, 
                ema_decay=decay_rate
            )

    def _update_ema(
            self, 
            target_params, 
            source_params, 
            ema_decay=.995
        ):
        """
        Update target parameters to be closer to those of source parameters using
        an exponential moving average.

        :param target_params: the target parameter sequence.
        :param source_params: the source parameter sequence.
        :param ema_decay: the EMA decay rate (closer to 1 means slower).
        """
        for targ, src in zip(target_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        lr = self.train_lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
    
    def _load_params(self):
        # logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
        self.edm_diffusion.diffusion_model.load_state_dict(
            torch.load(self.resume_diffusion_checkpoint)
        )

    def _load_ema_params(self, decay_rate):
        ema_params = copy.deepcopy(self.master_params)

        if self.resume_diffusion_checkpoint:
            # logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = torch.load(self.resume_diffusion_checkpoint)
            ema_params = self.state_dict_to_master_params(state_dict)
        else:
            raise ValueError("The EMA checkpoint does not exist.")
        
        return ema_params

    def _load_optimizer_state(self):
        opt_checkpoint = blobfile.join(
            blobfile.dirname(self.logdir), f"optimizer_{self.resume_epoch:04}.pt"
        )
        if blobfile.exists(opt_checkpoint):
            # logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = torch.load(opt_checkpoint)
            self.optimizer.load_state_dict(state_dict)
            print(f'[ utils/training ] Loaded optimizer state from {opt_checkpoint}')


class EDMPolicyTrainer(EDMTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train(self, summary_writer):
        timer = Timer()
        min_val_loss = float('inf')

        num_epochs = int(self.num_train_steps // len(self.train_dataset))

        for epoch in range(self.resume_epoch + 1, self.resume_epoch + num_epochs + 1):
            train_loss_total = 0
            num_train_batches = 0

            self.train_dataloader = torch.utils.data.DataLoader(
                self.train_dataset, 
                batch_size=self.train_batch_size, 
                num_workers=0, 
                shuffle=True, 
                collate_fn=self.train_dataset.collate_fn, 
                pin_memory=True
            )
            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset, 
                batch_size=self.val_batch_size, 
                num_workers=0, 
                shuffle=False, 
                collate_fn=self.val_dataset.collate_fn, 
                pin_memory=True
            )
            
            # Training loop
            for train_batch in self.train_dataloader:
                train_batch = batch_to_device(train_batch)
                train_loss = self.backward_loss(train_batch, train=True)
                train_loss_total += train_loss.item()
                num_train_batches += 1

                took_step = self.optimize()
                if took_step:
                    self.update_ema()
                    self.step += 1
                self._anneal_lr()

                if self.step < 2e5 and not self.step % self.log_freq:
                    summary_writer.add_scalar('step_loss/train_policy', train_loss, self.step)

            if not num_train_batches:
                raise ValueError(
                    "Empty training batch found. Check DataLoader and Dataset."
                )
            # Compute mean training loss for the epoch
            train_loss_mean = train_loss_total / num_train_batches

            # Validation loop
            val_loss_total = 0
            num_val_batches = 0
            with torch.no_grad():
                for val_batch in self.val_dataloader:
                    val_batch = batch_to_device(val_batch)
                    val_loss = self.backward_loss(val_batch, train=False)
                    val_loss_total += val_loss.item()
                    num_val_batches += 1

            if not num_val_batches:
                raise ValueError(
                    "Empty validation batch found. Check DataLoader and Dataset."
                )
            # Compute mean validation loss for the epoch
            val_loss_mean = val_loss_total / num_val_batches

            if val_loss_mean < min_val_loss:
                min_val_loss = val_loss_mean
                self.save(epoch)

            print(f'Epoch {epoch:4d} train_loss_policy: {train_loss_mean:8.4f} | val_loss_policy: {val_loss_mean:8.4f} | min_val_loss_policy: {min_val_loss:8.4f} | t: {timer():8.4f}')
            summary_writer.add_scalar('epoch_loss/train_policy', train_loss_mean, epoch)
            summary_writer.add_scalar('epoch_loss/valid_policy', val_loss_mean, epoch)
            summary_writer.flush()
                
    def backward_loss(self, batch, train=True):
        self._zero_grad()
        x = batch.actions
        cond = batch.conditions
        
        t, weights = self.schedule_sampler.sample(
            x.shape[0], 
            device=x.device
        )
        loss, _ = self.edm_diffusion.loss(
            x_start=x, 
            sigma=t, 
            cond=cond
        )

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, loss.detach()
            )
        loss = (loss * weights).mean()
        if not train: return loss

        self.backward(loss)

        return loss

    def save(self, epoch, ema_decay=None, ema_params=None):
        def save_diffusion_checkpoint(ema_decay, params):
            diffusion_model_state_dict = master_params_to_state_dict(
                self.edm_diffusion.diffusion_model, 
                self.param_groups_and_shapes, 
                params, 
                self.use_fp16
            )
            if ema_decay:
                savepath = os.path.join(
                    self.logdir, 
                    f'policy_ema_{ema_decay}_state_{(epoch):04d}.pt'
                )
            else:
                savepath = os.path.join(
                    self.logdir, 
                    f'policy_state_{(epoch):04d}.pt'
                )
            
            torch.save(diffusion_model_state_dict, savepath)
            print(f'[ utils/training ] Saved model to {savepath}')

        def save_point_cloud_projection_checkpoint(params):
            point_cloud_projector_state_dict = master_params_to_state_dict(
                self.edm_diffusion.point_cloud_to_2d_projector, 
                self.param_groups_and_shapes, 
                params, 
                self.use_fp16
            )
            savepath = os.path.join(
                self.logdir, 
                f'point_to_2d_{(epoch):04d}.pt'
            )
            torch.save(point_cloud_projector_state_dict, savepath)
            print(f'[ utils/training ] Saved model to {savepath}')

        def save_optimizer_checkpoint():
            savepath = os.path.join(
                self.logdir, 
                f'optimizer_{(epoch):04d}.pt'
            )
            torch.save(self.optimizer.state_dict(), savepath)
            print(f'[ utils/training ] Saved optimizer state to {savepath}')

        if ema_decay is None: ema_decay = self.ema_decay
        if ema_params is None: ema_params = self.ema_params
        for decay, params in zip(ema_decay, ema_params):
            save_diffusion_checkpoint(decay, params)
        if self.resume_point_cloud_checkpoint is None \
            and self.resume_flat_map_checkpoint is None and \
                self.point_cloud_to_2d_params is not None:
            save_point_cloud_projection_checkpoint(self.point_cloud_to_2d_params)
        save_optimizer_checkpoint()
