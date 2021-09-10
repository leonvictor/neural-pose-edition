import torch
import torch.nn.functional as F

from npe.networks.dataset import TargetInClipDataset
from npe.networks.utils import load_config
from npe.networks.ae.model import PoseAutoEncoder
from .model import IKModel


config = load_config('./config.yaml')

device = torch.device(
    "cuda:0" if torch.cuda.is_available() and config['use_cuda']
    else "cpu")

dataset = TargetInClipDataset(
    files=config['datasets'],
    folder=config['directory'],
    target_joints_names=config['targets']
)

data_loader = torch.utils.data.DataLoader(
    dataset,
    shuffle=True,
    batch_size=config['batch_size'],
    drop_last=True
)

ae = PoseAutoEncoder.from_weights(config['ae_weights']).to(device)

model = IKModel(latent_size=config['latent_dim'], targets=config['targets']).to(device)

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

def loss_fn(expected_pose, result_pose, targets_indices):
    """
    :param start: latent repr. of starting pose
    :param end: latent repr. of matched pose
    :parm res: network output (expected: diff with start to move towards end)
    :return: 
    """

    dists = F.mse_loss(expected_pose, result_pose, reduction='none')
    
    # weight losses (hands vs rest of the pose)
    weight = 100
    dists[..., targets_indices] *= weight
    return dists.mean()


epochs = 10
for epoch in range(epochs):
    for batch, x in enumerate(data_loader):
        optim.zero_grad()

        starting_pose = x['starting_pose'].float().to(device)
        targets = x['targets'].float().to(device)
        t2 = x['target_pose'][..., dataset.targets_indices]
        starting_lat = ae.encode(starting_pose)
        res = model(starting_lat, targets)
        res = ae.decode(res)

        loss = loss_fn(x['target_pose'].float().to(device), res, dataset.targets_indices)

        loss.backward()
        optim.step()

        print('Epoch {} [{}/{} ({:.0f}%)] Loss: {:.4f}'.format(
            epoch, batch+1, len(data_loader),
            (batch + 1) * 100 / len(data_loader),
            loss.item()))

    torch.save(model.state_dict(), config['constraints_weights'])
