from dataclasses import dataclass
import argparse

from tetriserve.xfuser.logger import init_logger
from tetriserve.xfuser.config import EngineConfig
from tetriserve.xfuser.config.args import xFuserArgs
from torch.distributed.argparse_util import env

logger = init_logger(__name__)


@dataclass
class ServerArgs:
    """Arguments for cFuser server."""

    nnodes: int = 1
    nproc_per_node: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 1037
    schedule_logic: str = "naive"
    enable_tracer: bool = False
    log_requests: bool = False
    perf_model_dir: str = "log/benchmark/scaling_efficiency"
    fake_engine: bool = (
        False  # if true, the engine will not run the actual inference, but only sleep for estimated time and return a fake output, this is used for testing the async engine and scheduler performance
    )
    skip_latent_transfer: bool = (
        False  # if true, skip real NCCL latent transfers between rounds and use dummy tensors instead. Denoising computation is still real. Useful for benchmarking scheduling without NCCL transfer bugs.
    )

    engine_config: EngineConfig = None

    def __post_init__(self):
        # Multi-node requires cross-node NCCL setup + distributed latent transfer,
        # which is not yet implemented. Only single-node multi-GPU is supported.
        assert self.nnodes == 1, "Only support single node server"
        if (
            self.engine_config.parallel_config.sp_config.ulysses_degree
            * self.engine_config.parallel_config.sp_config.ring_degree
            != self.nproc_per_node
        ):
            logger.warning(f"nproc_per_node is not equal to ulysses_degree * ring_degree")
            logger.warning(f"reset ulysses_degree to {self.nproc_per_node}, ring_degree to 1")
            self.engine_config.parallel_config.sp_config.ulysses_degree = self.nproc_per_node
            self.engine_config.parallel_config.sp_config.ring_degree = 1
            self.engine_config.parallel_config.sp_degree = self.nproc_per_node
            self.engine_config.parallel_config.ulysses_degree = self.nproc_per_node
            self.engine_config.parallel_config.ring_degree = 1
            assert (
                self.engine_config.parallel_config.sp_config.ulysses_degree
                * self.engine_config.parallel_config.sp_config.ring_degree
                == self.nproc_per_node
            ), "Failed to reset ulysses_degree and ring_degree"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        # Add server-specific arguments
        parser.add_argument(
            "--nnodes",
            type=str,
            default="1",
            help="Number of nodes",
        )
        parser.add_argument(
            "--nproc-per-node",
            "--nproc_per_node",
            type=str,
            default="1",
            help="Number of workers per node; supported values: [auto, cpu, gpu, int].",
        )
        parser.add_argument(
            "--master-addr",
            "--master_addr",
            default="127.0.0.1",
            type=str,
            action=env,
            help="Address of the master node (rank 0) that only used for static rendezvous. It should "
            "be either the IP address or the hostname of rank 0. For single node multi-proc training "
            "the --master-addr can simply be 127.0.0.1; IPv6 should have the pattern "
            "`[0:0:0:0:0:0:0:1]`.",
        )
        parser.add_argument(
            "--master-port",
            "--master_port",
            default=1037,
            type=int,
            action=env,
            help="Port on the master node (rank 0) to be used for communication during distributed "
            "training. It is only used for static rendezvous.",
        )
        # Note: this is called "schedule_logic" in CLI/code, "Scheduler" in docs,
        # and "schedule_logics" (plural list) in benchmark YAML configs.
        parser.add_argument(
            "--schedule_logic",
            type=str,
            choices=[
                "data_parallel",
                "fixed_sp_2",
                "fixed_sp_4",
                "fixed_sp_8",
                "fcfs_optimal_shape",
                "dyn_slo_schedule_global",
                "dyn_slo_schedule_global_scale_up",
                "dyn_slo_schedule_global_dp",
                "dyn_sp_schedule",
                "per_resolution_schedule",
                "per_resolution_schedule_1_1_4_8",
                "per_resolution_schedule_1_1_2_8",
            ],
            default="naive",
            help="Scheduler to use",
        )
        parser.add_argument(
            "--fake_engine",
            action="store_true",
            default=False,
            help="If true, the engine will not run the actual inference, but only sleep for estimated time and return a fake output, this is used for testing the async engine and scheduler performance",
        )
        parser.add_argument(
            "--skip_latent_transfer",
            action="store_true",
            default=False,
            help="If true, skip real NCCL latent transfers and use dummy tensors. Denoising is still real. For benchmarking scheduling without transfer bugs.",
        )
        parser.add_argument(
            "--enable_tracer",
            action="store_true",
            default=False,
            help="If true, the engine will export a trace file",
        )
        parser.add_argument(
            "--log_requests",
            action="store_true",
            default=False,
            help="If true, the engine will log the requests",
        )
        parser.add_argument(
            "--perf_model_dir",
            type=str,
            default="log/benchmark/scaling_efficiency",
            help="The directory to save the performance model",
        )

        # Add engine-related arguments through cFuserArgs
        parser = xFuserArgs.add_cli_args(parser)

        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        engine_args = xFuserArgs.from_cli_args(args)
        engine_config, _ = engine_args.create_config()
        return cls(
            nnodes=int(args.nnodes),
            nproc_per_node=int(args.nproc_per_node),
            master_addr=args.master_addr,
            master_port=int(args.master_port),
            schedule_logic=args.schedule_logic,
            engine_config=engine_config,
            fake_engine=args.fake_engine,
            skip_latent_transfer=args.skip_latent_transfer,
            enable_tracer=args.enable_tracer,
            log_requests=args.log_requests,
            perf_model_dir=args.perf_model_dir,
        )
