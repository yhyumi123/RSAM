from copy import deepcopy
from pathlib import Path
import pytest
import torch
import yaml
from rsam.train import CycleGenerators, validate_training_config
from rsam.critic import ct_critic_input
from rsam.anatomy_losses import body_reconstruction_loss


def config():
    return yaml.safe_load((Path(__file__).parents[1]/'configs/rsam.yaml').read_text())


def test_nce_parameters_receive_updates_without_changing_generator_export():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    cfg=config()['model'];cfg.update(base_channels=8,hidden_dim=8,image_size=32,n_residual_blocks=3)
    network=CycleGenerators(cfg,nce_enabled=True)
    optimizer=torch.optim.Adam(network.parameters(),lr=.001)
    before=network.projection_heads.heads[0][0].weight.detach().clone()
    output=network(torch.rand(2,3,32,32)*2-1,torch.rand(2,3,32,32)*2-1,torch.eye(3)[:2])
    output['nce_loss'].backward()
    assert torch.isfinite(output['nce_loss'])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.projection_heads.parameters())
    optimizer.step()
    assert not torch.equal(before,network.projection_heads.heads[0][0].weight)
    assert not any('projection' in k for k in network.mr_to_ct.state_dict())


def test_critic_mask_preserves_other_cases_and_input_and_backward():
    image=torch.randn(3,3,32,32,requires_grad=True)
    original=image.detach().clone()
    body=torch.zeros_like(image,dtype=torch.bool);body[:,:,8:24,8:24]=True
    batch={'paired':torch.tensor([True,False,False]),'code':torch.eye(3),'mr_body':body,'ct_body':body}
    value=ct_critic_input(image,batch,'fake',[2])
    assert torch.equal(image.detach(),original)
    assert torch.equal(value[:2],image[:2])
    assert torch.equal(value,ct_critic_input(image,batch,'real',[2]))
    value.sum().backward()
    assert image.grad[2][~body[2]].abs().sum()==0
    assert image.grad[2][body[2]].abs().sum()>0
    batch['mr_body']=torch.zeros_like(body)
    assert (ct_critic_input(image,batch,'fake',[2])[2]==-1).all()


def test_disabled_background_does_not_affect_body_reconstruction():
    pred=torch.zeros(1,3,16,16,requires_grad=True)
    body=torch.zeros_like(pred,dtype=torch.bool);body[:,:,4:12,4:12]=True
    target=torch.where(body,torch.zeros_like(pred),torch.ones_like(pred))
    off=body_reconstruction_loss(pred,target,torch.tensor([True]),body,background_weight=0)
    on=body_reconstruction_loss(pred,target,torch.tensor([True]),body,background_weight=.1)
    assert off.item()==0 and on.item()>0


def test_crop_cannot_use_replay_with_unmatched_masks():
    cfg=config();validate_training_config(cfg)
    cfg['training']['replay_size']=10
    with pytest.raises(ValueError,match='matching body masks'):validate_training_config(cfg)
