import torch
import torch.nn.functional as F

from .model import PoseAutoEncoder

from npe.networks.dataset import PoseDataset
from npe.networks.utils import load_config
from npe.networks.loss import joint_distance

config = load_config('./config.yaml')

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and config['use_cuda']
    else "cpu")

dataset = PoseDataset(
    config['datasets'],
    config['directory'],
    save_stats="stats",
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=config['batch_size'],
    drop_last=True
)

ae = PoseAutoEncoder(
    dataset.input_dim, config['latent_dim'], share_weights=config['share_weights']).to(device)

ae_optim = torch.optim.Adam(
    ae.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

for epoch in range(30):
    for batch, x in enumerate(data_loader):
        global_step = int(epoch * len(data_loader) + batch)

        clips = x['clip'].float().to(device)

        ae_optim.zero_grad()

        z = ae.encode(clips)
        recon = ae.decode(z)

        # reconstruction loss
        loss = F.mse_loss(clips, recon)
        # loss = joint_distance(recon, clips)

        loss.backward()
        ae_optim.step()

        print('Epoch {} [{}/{} ({:.0f}%)] loss: {:.4f}'.format(
            epoch, batch+1, len(data_loader),
            (batch + 1) * 100 / len(data_loader),
            loss.item()))

    torch.save(ae.state_dict(),
               config['ae_weights'])
