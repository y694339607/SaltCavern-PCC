import torch


def chamfer_distance_pure(pcs1, pcs2):
    B, N, _ = pcs1.shape
    _, M, _ = pcs2.shape

    dist = torch.cdist(pcs1, pcs2, p=2)

    dist1 = torch.min(dist, dim=2)[0]
    dist2 = torch.min(dist, dim=1)[0]

    return dist1, dist2


def cd_loss_L1(pcs1, pcs2):
    dist1, dist2 = chamfer_distance_pure(pcs1, pcs2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcs1, pcs2):
    dist1, dist2 = chamfer_distance_pure(pcs1, pcs2)
    return torch.mean(dist1 ** 2) + torch.mean(dist2 ** 2)


def emd_loss(pcs1, pcs2):
    from extensions.earth_movers_distance.emd import EarthMoverDistance
    EMD = EarthMoverDistance()
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists)
