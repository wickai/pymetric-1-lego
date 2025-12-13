import torch


def ema_avg_fn(averaged_model_parameter, model_parameter, num_averaged):
    # EMA 的 decay 值；建议你根据 batch size 调整
    decay = 0.9999
    return decay * averaged_model_parameter + (1. - decay) * model_parameter

class EmaDDPWrapper:
    def __init__(self, model, device_ids, output_device):
        # Keep original AveragedModel for update
        self.ema_model = model
        # Wrap internal model with DDP for eval
        self.ddp_model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=device_ids,
            output_device=output_device,
            find_unused_parameters=False,
        )

    def update_parameters(self, model):
        # Forward update call to the original EMA model
        self.ema_model.update_parameters(model)

    def forward(self, *args, **kwargs):
        return self.ddp_model(*args, **kwargs)

    def eval(self):
        self.ddp_model.eval()
        
    def train(self):
        self.ddp_model.train()

    def to(self, device):
        self.ema_model.to(device)
        self.ddp_model.to(device)

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)
        
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)  # 转发调用
