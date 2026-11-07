import torch
import open3d as o3d


def chamfer_distance_pure(pcs1, pcs2):
    B, N, _ = pcs1.shape
    _, M, _ = pcs2.shape

    dist = torch.cdist(pcs1, pcs2, p=2)

    dist1 = torch.min(dist, dim=2)[0]
    dist2 = torch.min(dist, dim=1)[0]

    return dist1, dist2


def l1_cd(pcs1, pcs2):
    dist1, dist2 = chamfer_distance_pure(pcs1, pcs2)
    dist1 = torch.mean(dist1, dim=1)
    dist2 = torch.mean(dist2, dim=1)
    return torch.sum(dist1 + dist2) / 2


def l2_cd(pcs1, pcs2):
    dist1, dist2 = chamfer_distance_pure(pcs1, pcs2)
    dist1 = torch.mean(dist1 ** 2, dim=1)
    dist2 = torch.mean(dist2 ** 2, dim=1)
    return torch.sum(dist1 + dist2)


def emd(pcs1, pcs2):
    from extensions.earth_movers_distance.emd import EarthMoverDistance
    EMD = EarthMoverDistance()
    dists = EMD(pcs1, pcs2)
    return torch.sum(dists)


def f_score(pred, gt, th=0.01):
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0
