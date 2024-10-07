from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool

pool_registry = dict(
    mean=global_mean_pool,
    sum=global_add_pool,
    max=global_max_pool,
)
