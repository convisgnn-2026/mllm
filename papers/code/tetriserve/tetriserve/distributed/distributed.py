from tetriserve.xfuser.core.distributed.parallel_state import (
    init_distributed_environment as xfuser_init_distributed_environment,
)


def init_distributed_environment(
    local_rank: int, rank: int, world_size: int, distributed_init_method: str, backend: str
):
    xfuser_init_distributed_environment(
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        distributed_init_method=distributed_init_method,
        backend=backend,
    )
