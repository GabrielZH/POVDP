import numpy as np


class EMAScalesFunction(object):
    def __init__(
            self, 
            target_ema_mode, 
            start_ema, 
            scale_mode, 
            start_scales, 
            end_scales, 
            total_steps, 
            distill_steps_per_iter
    ):
        self.target_ema_mode = target_ema_mode
        self.start_ema = start_ema
        self.scale_mode = scale_mode
        self.start_scales = start_scales
        self.end_scales = end_scales
        self.total_steps = total_steps
        self.distill_steps_per_iter=distill_steps_per_iter

    def ema_scales_fn(self, step):
        if self.target_ema_mode == "fixed" and self.scale_mode == "fixed":
            target_ema = self.start_ema
            scales = self.start_scales
        elif self.target_ema_mode == "fixed" and self.scale_mode == "progressive":
            target_ema = self.start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / self.total_steps) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
                    + self.start_scales**2
                ) - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif self.target_ema_mode == "adaptive" and self.scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / self.total_steps) * ((self.end_scales + 1) ** 2 - self.start_scales**2)
                    + self.start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(self.start_ema) * self.start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)