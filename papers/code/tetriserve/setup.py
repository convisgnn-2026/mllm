from setuptools import find_packages, setup, Extension
import subprocess
import platform
import os
import sys
import glob


def get_cuda_version():
    try:
        nvcc_version = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        version_line = [line for line in nvcc_version.split("\n") if "release" in line][0]
        cuda_version = version_line.split(" ")[-2].replace(",", "")
        return "cu" + cuda_version.replace(".", "")
    except Exception as e:
        return "no_cuda"


def clean_cpp_extensions():
    """Clean up existing C++ extension files to force rebuild."""
    patterns_to_clean = [
        "tetriserve/scheduler/cpp_optimizer/cpp_optimizer*.so",
        "tetriserve/scheduler/cpp_optimizer/*.so",
        "build/lib*/tetriserve/scheduler/cpp_optimizer/*.so",
        "build/temp*/**/cpp_optimizer*",
    ]

    for pattern in patterns_to_clean:
        for file_path in glob.glob(pattern, recursive=True):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                elif os.path.isdir(file_path):
                    import shutil

                    shutil.rmtree(file_path)
                    print(f"Removed directory: {file_path}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")


def get_cpp_extension():
    """Build C++ optimizer extension if possible."""
    try:
        # Import pybind11 modules only when needed
        from pybind11.setup_helpers import Pybind11Extension

        # Clean existing extensions first
        clean_cpp_extensions()

        # Try to use local pybind11 first, then fallback to system installation
        pybind11_include = None
        try:
            # Add local pybind11 to path
            local_pybind11_path = os.path.join(os.path.dirname(__file__), "3rdparty", "pybind11")
            if os.path.exists(local_pybind11_path):
                sys.path.insert(0, local_pybind11_path)
                import pybind11
                from pybind11 import get_include as pybind11_get_include

                pybind11_include = pybind11_get_include()
                print(f"Using local pybind11 from: {local_pybind11_path}")
            else:
                # Fallback to system pybind11
                import pybind11
                from pybind11 import get_include as pybind11_get_include

                pybind11_include = pybind11_get_include()
                print("Using system pybind11")
        except ImportError:
            print("Warning: pybind11 not found, C++ optimizer will not be available")
            return None

        # Compiler flags
        cpp_args = ["-std=c++14"]
        link_args = []

        if platform.system() == "Darwin":  # macOS
            cpp_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
        elif platform.system() == "Windows":
            cpp_args = ["/std:c++14"]
        else:  # Linux
            cpp_args += ["-fPIC"]

        # Optimization flags
        if os.environ.get("DEBUG") != "1":
            if platform.system() == "Windows":
                cpp_args += ["/O2", "/DNDEBUG"]
            else:
                cpp_args += ["-O3", "-DNDEBUG", "-funroll-loops"]
        else:
            cpp_args += ["-g", "-DDEBUG"]

        # Include directories
        include_dirs = [pybind11_include]

        # Add local pybind11 include if using local version
        local_pybind11_include = os.path.join(os.path.dirname(__file__), "3rdparty", "pybind11", "include")
        if os.path.exists(local_pybind11_include):
            include_dirs.append(local_pybind11_include)

        return Pybind11Extension(
            "tetriserve.scheduler.cpp_optimizer.cpp_optimizer",
            sources=[
                "tetriserve/scheduler/cpp_optimizer/pybind_wrapper.cpp",
            ],
            include_dirs=include_dirs,
            language="c++",
            cxx_std=14,
            extra_compile_args=cpp_args,
            extra_link_args=link_args,
        )
    except Exception as e:
        print(f"Warning: Failed to build C++ optimizer: {e}")
        return None


if __name__ == "__main__":
    with open("README.md", "r") as f:
        long_description = f.read()
    fp = open("tetriserve/__version__.py", "r").read()
    version = eval(fp.strip().split()[-1])

    # C++ optimizer extension is disabled. The Python fallback in
    # dyn_step_scheduler.py is functionally equivalent. To re-enable:
    #   1. Install pybind11: pip install pybind11
    #   2. Uncomment the lines below
    #   3. Run: pip install -e . (will compile the C++ extension)
    ext_modules = []
    cmdclass = {}
    # cpp_ext = get_cpp_extension()
    # if cpp_ext:
    #     ext_modules = [cpp_ext]
    #     from pybind11.setup_helpers import build_ext
    #     cmdclass = {"build_ext": build_ext}

    setup(
        name="tetriserve",
        author="TetriServe Team",
        author_email="tetriserve@gmail.com",
        packages=find_packages(),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        install_requires=[
            "torch>=2.1.0",
            "accelerate>=0.33.0",
            "transformers>=4.39.1",
            "sentencepiece>=0.1.99",
            "distvae",
            "pytest",
            "einops",
            "pandas",
            "pyzmq",
            "protobuf",
            "numpy<2",
            "diffusers==0.33.0",
        ],
        extras_require={
            "flash-attn": [
                "flash-attn==2.6.3",
            ],
        },
        url="https://github.com/DiT-Serving/TetriServe",
        description="TetriServe: A deadline-aware diffusion transformer serving system with step-level dynamic sequence parallelism",
        long_description=long_description,
        long_description_content_type="text/markdown",
        version=version,
        classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
        include_package_data=True,
        python_requires=">=3.10",
    )
