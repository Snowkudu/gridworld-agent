import torch

def to_cnn_1ch(
    states:torch.Tensor,
    height: int =10,
    width: int =10,
        
) -> torch.Tensor:
    if states.ndim !=2:
        raise ValueError(f"Exepcted [N,features],got a shape of {tuple(states.shape)}")
    expected_features= height*width
    if states.shape[1] != expected_features:
        raise ValueError(f"Exepcted {expected_features},got {states.shape[1]}")
    return states.reshape(-1,1,height,width)

def to_cnn_3ch(
    states:torch.Tensor,
    height: int =10,
    width: int =10, 
)->torch.Tensor:
    if states.ndim !=2:
        raise ValueError(f"Exepcted [N,features],got a shape of {tuple(states.shape)}")
    expected_features= height*width
    if states.shape[1] != expected_features:
        raise ValueError(f"Exepcted {expected_features},got {states.shape[1]}")
    grids= states.reshape(-1,height,width)
    obstacles = grids == -1
    agents = grids == 1
    goals = grids == 2
    x=torch.stack((obstacles,agents,goals),dim=1)
    return x.to(dtype=states.dtype)