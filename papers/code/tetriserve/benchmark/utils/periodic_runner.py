#!/usr/bin/env python3
"""
Periodic Command Runner

This script periodically:
1. Kills all Python processes (except itself)
2. Runs a specified command
3. Repeats every 10 minutes

Usage:
    python periodic_runner.py "your command here"
    python periodic_runner.py --interactive
    python periodic_runner.py --help
"""

import argparse
import subprocess
import time
import signal
import sys
import os
from datetime import datetime
from typing import Optional


class PeriodicRunner:
    def __init__(self, command: str, interval_minutes: int = 10, safe_mode: bool = True, enforce_timeout: bool = True):
        self.command = command
        self.interval_seconds = interval_minutes * 60
        self.safe_mode = safe_mode
        self.enforce_timeout = enforce_timeout
        self.running = True
        self.current_script_pid = os.getpid()

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and termination signals gracefully."""
        print(f"\n[{self._timestamp()}] Received signal {signum}. Shutting down gracefully...")
        self.running = False

    def _timestamp(self) -> str:
        """Get current timestamp for logging."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log(self, message: str):
        """Log a message with timestamp."""
        print(f"[{self._timestamp()}] {message}")

    def _kill_python_processes(self) -> bool:
        """
        Kill all Python processes except this script.
        Returns True if successful, False otherwise.
        """
        try:
            if self.safe_mode:
                # Get list of python processes first
                result = subprocess.run(["pgrep", "-f", "python"], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    pids = result.stdout.strip().split("\n")
                    pids = [pid for pid in pids if pid and int(pid) != self.current_script_pid]

                    if pids:
                        self._log(f"Found {len(pids)} Python processes to kill (excluding this script)")
                        for pid in pids:
                            try:
                                subprocess.run(["kill", pid], timeout=5)
                            except subprocess.TimeoutExpired:
                                self._log(f"Warning: Timeout killing PID {pid}")
                    else:
                        self._log("No Python processes found to kill")
                else:
                    self._log("No Python processes found")
            else:
                # Less safe but more thorough - kill all python processes
                self._log("Killing all Python processes...")
                result = subprocess.run(["pkill", "-f", "python"], capture_output=True, text=True, timeout=10)

                if result.returncode == 0:
                    self._log("Successfully killed Python processes")
                elif result.returncode == 1:
                    self._log("No Python processes found to kill")
                else:
                    self._log(f"pkill returned code {result.returncode}")

            # Wait for processes to fully terminate
            time.sleep(3)
            return True

        except subprocess.TimeoutExpired:
            self._log("Error: Timeout while killing Python processes")
            return False
        except Exception as e:
            self._log(f"Error killing Python processes: {e}")
            return False

    def _run_command(self) -> bool:
        """
        Run the specified command with timeout enforcement.
        If command takes longer than the interval, it will be killed.
        Returns True if successful, False otherwise.
        """
        try:
            self._log(f"Running command: {self.command}")

            # Set timeout based on enforce_timeout setting
            if self.enforce_timeout:
                timeout_seconds = self.interval_seconds
                self._log(f"Command timeout set to {timeout_seconds} seconds ({self.interval_seconds // 60} minutes)")
            else:
                timeout_seconds = None
                self._log("No timeout set - command will run until completion")

            # Run command with optional timeout
            result = subprocess.run(
                self.command,
                shell=True,
                capture_output=False,  # Let output go to console
                text=True,
                timeout=timeout_seconds,  # Timeout = interval (if enforce_timeout=True)
            )

            if result.returncode == 0:
                self._log("Command completed successfully")
                return True
            else:
                self._log(f"Command failed with return code {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            if self.enforce_timeout:
                self._log(
                    f"⚠️  Command exceeded {timeout_seconds // 60} minute timeout - killing and starting next cycle"
                )
            else:
                self._log("Error: Command timed out unexpectedly")
            return False
        except KeyboardInterrupt:
            self._log("Command interrupted by user")
            return False
        except Exception as e:
            self._log(f"Error running command: {e}")
            return False

    def run_cycle(self) -> bool:
        """
        Run one complete cycle (kill + command).
        Returns True if successful, False otherwise.
        """
        self._log("=" * 60)
        self._log("Starting new cycle")

        # Step 1: Kill Python processes
        if not self._kill_python_processes():
            self._log("Failed to kill Python processes, but continuing...")

        # Step 2: Run the command
        success = self._run_command()

        self._log(f"Cycle completed {'successfully' if success else 'with errors'}")
        return success

    def start(self):
        """Start the periodic runner."""
        self._log("=" * 60)
        self._log(f"Starting Periodic Runner")
        self._log(f"Command: {self.command}")
        self._log(f"Interval: {self.interval_seconds // 60} minutes")
        self._log(f"Safe mode: {self.safe_mode}")
        if self.enforce_timeout:
            self._log("⚠️  Commands will be killed if they exceed the interval duration")
        else:
            self._log("ℹ️  Commands will run until completion (no timeout enforcement)")
        self._log("Press Ctrl+C to stop")
        self._log("=" * 60)

        cycle_count = 0

        try:
            while self.running:
                cycle_count += 1
                self._log(f"Starting cycle {cycle_count}")

                # Record cycle start time
                cycle_start_time = time.time()

                # Run one cycle (includes timeout enforcement)
                self.run_cycle()

                if not self.running:
                    break

                # Calculate actual cycle duration
                cycle_duration = time.time() - cycle_start_time
                self._log(f"Cycle {cycle_count} took {cycle_duration:.1f} seconds")

                # If cycle completed faster than interval, wait for the remainder
                if cycle_duration < self.interval_seconds:
                    remaining_time = self.interval_seconds - cycle_duration
                    self._log(f"Waiting {remaining_time:.1f} seconds until next cycle...")

                    # Sleep in chunks to allow for graceful shutdown
                    sleep_chunks = int(remaining_time // 10)  # Check every 10 seconds
                    for i in range(sleep_chunks):
                        if not self.running:
                            break
                        time.sleep(10)

                    # Sleep remaining time
                    final_sleep = remaining_time % 10
                    if final_sleep > 0 and self.running:
                        time.sleep(final_sleep)
                else:
                    # Cycle took the full interval (likely due to timeout), start next cycle immediately
                    self._log("🔄 Starting next cycle immediately (previous cycle used full interval)")

        except KeyboardInterrupt:
            self._log("Interrupted by user")
        except Exception as e:
            self._log(f"Unexpected error: {e}")
        finally:
            self._log("Periodic Runner stopped")
            self._log(f"Total cycles completed: {cycle_count}")


def get_interactive_command() -> str:
    """Get command interactively from user."""
    print("=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter the command you want to run periodically.")
    print("This command will be executed every 10 minutes after killing Python processes.")
    print()

    while True:
        command = input("Command: ").strip()
        if command:
            print(f"\nYou entered: {command}")
            confirm = input("Is this correct? (y/n): ").strip().lower()
            if confirm in ["y", "yes"]:
                return command
            else:
                print("Please enter the command again.")
        else:
            print("Please enter a valid command.")


def main():
    parser = argparse.ArgumentParser(
        description="Periodically kill Python processes and run a command",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a benchmark every 10 minutes
  python periodic_runner.py "python benchmark_serving.py --config test.yaml"
  python benchmark/serving/periodic_runner.py "mkdir -p /path/to/results/dyn_global_benchmark_H100_sd3 && \
  python benchmark/serving/benchmark_serving.py --config benchmark/serving/configs/dyn_global_benchmark_H100_sd3_node1.yaml 2>&1 | tee /path/to/results/dyn_global_benchmark_H100_sd3/benchmark_node1.log"

  # Interactive mode
  python periodic_runner.py --interactive

  # Custom interval and unsafe mode
  python periodic_runner.py "echo hello" --interval 5 --unsafe
        """,
    )

    parser.add_argument("command", nargs="?", help="Command to run periodically (required unless --interactive)")

    parser.add_argument("--interactive", action="store_true", help="Enter command interactively")

    parser.add_argument("--interval", type=int, default=10, help="Interval between runs in minutes (default: 10)")

    parser.add_argument("--unsafe", action="store_true", help="Use pkill -f python (less safe but more thorough)")

    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without actually doing it")

    parser.add_argument(
        "--no-timeout", action="store_true", help="Don't enforce timeout - let commands run until completion"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.interactive and not args.command:
        parser.error("Either provide a command or use --interactive mode")

    if args.interval < 1:
        parser.error("Interval must be at least 1 minute")

    # Get command
    if args.interactive:
        command = get_interactive_command()
    else:
        command = args.command

    # Dry run mode
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN MODE")
        print("=" * 60)
        print(f"Would run command: {command}")
        print(f"Every {args.interval} minutes")
        print(f"Safe mode: {not args.unsafe}")
        print("No actual processes will be killed or commands executed.")
        return

    # Confirm if in unsafe mode
    if args.unsafe:
        print("=" * 60)
        print("WARNING: UNSAFE MODE")
        print("=" * 60)
        print("This will use 'pkill -f python' which may kill this script too!")
        confirm = input("Are you sure you want to continue? (y/n): ").strip().lower()
        if confirm not in ["y", "yes"]:
            print("Aborted.")
            return

    # Create and start runner
    runner = PeriodicRunner(
        command=command, interval_minutes=args.interval, safe_mode=not args.unsafe, enforce_timeout=not args.no_timeout
    )

    runner.start()


if __name__ == "__main__":
    main()
