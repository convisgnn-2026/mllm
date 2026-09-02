import zmq
import zmq.asyncio
import socket
from typing import List
from torch.multiprocessing.spawn import spawn
import pickle
import asyncio
import time

from tetriserve.xfuser.logger import init_logger

logger = init_logger(__name__)


def find_free_port():
    """Find a free port on the system.

    Returns:
        int: An available port number
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def find_free_ports(num_ports):
    """
    Find a list of consecutive free ports

    Args:
        num_ports (int): Number of consecutive ports needed

    Returns:
        list: List of consecutive free port numbers
    """
    import socket

    def is_port_free(port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return True
            except OSError:
                return False

    # Start checking from a common range (e.g., 8000)
    start_port = 8000
    max_port = 65535

    while start_port + num_ports <= max_port:
        ports = list(range(start_port, start_port + num_ports))
        if all(is_port_free(port) for port in ports):
            return ports
        start_port += 1

    raise RuntimeError(f"Could not find {num_ports} consecutive free ports")


class Router:
    def __init__(self, address: str, port: int, dealer_num: int):
        self.context = zmq.asyncio.Context()
        self.router = self.context.socket(zmq.ROUTER)

        # Modify socket options
        self.router.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.router.setsockopt(zmq.RCVHWM, 1000)  # Lower buffer size
        self.router.setsockopt(zmq.SNDHWM, 1000)  # Lower buffer size
        self.router.setsockopt(zmq.ROUTER_MANDATORY, 1)  # Raise error on unroutable messages

        self.router.bind(f"ipc://{address}:{port}")
        self.dealer_num = dealer_num
        self.workers = {}

    async def init(self):
        logger.info(f"Router starting initialization, waiting for {self.dealer_num} dealers...")
        logger.info(
            f"Router ready to accept connections on port {self.router.getsockopt_string(zmq.LAST_ENDPOINT) if hasattr(self.router, 'getsockopt_string') else 'unknown'}"
        )
        registered = 0

        while registered < self.dealer_num:
            try:
                message_parts = await self.router.recv_multipart()
                identity = message_parts[0]
                worker_id = identity.decode()
                msg = pickle.loads(message_parts[-1])

                if msg == "init":
                    if worker_id not in self.workers:
                        self.workers[worker_id] = identity
                        registered += 1
                        logger.info(f"Router registered worker {worker_id} ({registered}/{self.dealer_num})")
                        # Send immediate acknowledgment
                        await self.send_to_worker(worker_id, "init_ok")
            except Exception as e:
                logger.error(f"Error during router initialization: {str(e)}")
                await asyncio.sleep(0.1)  # Short delay before retry
                continue

        logger.info("Router initialization completed successfully")

    async def send_to_worker(self, worker_id: str, msg: object):
        if worker_id not in self.workers:
            raise ValueError(f"Worker {worker_id} not registered")
        try:
            identity = self.workers[worker_id]
            await self.router.send_multipart([identity, b"", pickle.dumps(msg)])
        except Exception as e:
            logger.error(f"Failed to send to worker {worker_id}: {str(e)}")
            raise

    async def recv_from_worker(self):
        message_parts = await self.router.recv_multipart()
        identity = message_parts[0]
        worker_id = identity.decode()
        # Skip the empty delimiter and return the payload
        return worker_id, pickle.loads(message_parts[-1])

    async def send_to_workers(self, worker_ids: List[str], msg: object):
        encoded_msg = pickle.dumps(msg)

        async def send_to_single_worker(worker_id: str):
            if worker_id not in self.workers:
                raise ValueError(f"Worker {worker_id} not registered")
            identity = self.workers[worker_id]
            try:
                await self.router.send_multipart([identity, b"", encoded_msg])
            except zmq.error.ZMQError as e:
                logger.error(f"Failed to send to worker {worker_id}: {str(e)}")
                raise

        await asyncio.gather(*[send_to_single_worker(worker_id) for worker_id in worker_ids])


class Dealer:
    def __init__(self, router_address: str, router_port: int, worker_id: str):
        self.context = zmq.Context()
        self.dealer = self.context.socket(zmq.DEALER)
        self.worker_id = worker_id

        # Modify socket options
        self.dealer.setsockopt(zmq.LINGER, 0)  # Don't wait on close
        self.dealer.setsockopt(zmq.RCVHWM, 1000)  # Lower buffer size
        self.dealer.setsockopt(zmq.SNDHWM, 1000)  # Lower buffer size
        self.dealer.setsockopt(zmq.RCVTIMEO, -1)  # No timeout, use non-blocking recv instead
        self.dealer.setsockopt(zmq.SNDTIMEO, 1000)  # 1 second send timeout

        self.dealer.identity = worker_id.encode()
        self.dealer.connect(f"ipc://{router_address}:{router_port}")

        logger.info(f"Dealer {worker_id} connecting to router...")
        self._initialize_connection()

    def _initialize_connection(self):
        # Add startup delay to avoid race condition with router initialization
        import time

        worker_rank = int(self.worker_id.split("-")[-1]) if "-" in self.worker_id else 0
        startup_delay = 2.0 + (worker_rank * 0.1)  # Staggered startup: 2.0, 2.1, 2.2, etc.
        logger.info(f"Dealer {self.worker_id} waiting {startup_delay:.1f}s for router startup...")
        time.sleep(startup_delay)

        max_retries = 10  # Increased retries since we're being more patient
        retry_count = 0

        while retry_count < max_retries:
            try:
                self.send_to_router("init")
                response = self.recv_from_router(timeout=2000)  # 2 second timeout for init

                if response == "init_ok":
                    logger.info(f"Dealer {self.worker_id} successfully registered with router")
                    return

            except zmq.error.Again:
                retry_count += 1
                logger.warning(f"Dealer {self.worker_id} retry {retry_count}/{max_retries}")
                time.sleep(1.0)  # Longer delay between retries for more patience
                continue

            except Exception as e:
                logger.error(f"Dealer {self.worker_id} failed to initialize: {str(e)}")
                raise

        raise RuntimeError(f"Dealer {self.worker_id} failed to initialize after {max_retries} retries")

    def send_to_router(self, msg: object):
        try:
            serialized_msg = pickle.dumps(msg)
            self.dealer.send_multipart([b"", serialized_msg], zmq.NOBLOCK)
        except zmq.error.Again:
            logger.warning(f"Dealer {self.worker_id} send would block, retrying...")
            time.sleep(0.1)
            self.dealer.send_multipart([b"", serialized_msg])  # Retry with blocking send
        except Exception as e:
            logger.error(f"Dealer {self.worker_id} failed to send message: {str(e)}")
            raise

    def recv_from_router(self, timeout=None):
        start_time = time.time()
        while True:
            try:
                if timeout is not None:
                    if time.time() - start_time > timeout / 1000:
                        raise zmq.error.Again("Timeout exceeded")

                # Try non-blocking receive first
                try:
                    response = self.dealer.recv_multipart(zmq.NOBLOCK)
                    return pickle.loads(response[-1])
                except zmq.error.Again:
                    time.sleep(0.01)  # Short sleep before retry
                    continue

            except Exception as e:
                if isinstance(e, zmq.error.Again):
                    raise  # Re-raise timeout errors
                logger.error(f"Dealer {self.worker_id} failed to receive message: {str(e)}")
                raise

    def cleanup(self):
        try:
            if hasattr(self, "dealer"):
                self.dealer.close(linger=0)
            if hasattr(self, "context"):
                self.context.term()
        except Exception as e:
            logger.error(f"Error during cleanup for {self.worker_id}: {str(e)}")


async def test_dealer_router(rank, world_size, router_address, router_port):
    if rank == 0:
        router = Router(router_address, router_port, world_size - 1)
        await router.init()

        for _ in range(router.dealer_num):
            worker_id, msg = await router.recv_from_worker()
            logger.info(f"router recv from worker {worker_id}: {msg}")
    else:
        identity = f"Worker-{rank}"

        dealer = Dealer(router_address, router_port, identity)

        dealer.send_to_router(f"{identity} pinging router")
        logger.info(f"{identity} ping sent to router")


def run_test_dealer_router(rank, world_size, router_address, router_port):
    asyncio.run(test_dealer_router(rank, world_size, router_address, router_port))


if __name__ == "__main__":
    world_size = 5
    router_port = find_free_port()

    spawn(
        run_test_dealer_router,
        nprocs=world_size,
        args=(world_size, "localhost", router_port),
    )
