from setuptools import setup, find_packages

with open("README.md", "r") as f:
    my_long_description = f.read()

with open('LICENSE') as f:
    my_license = f.read()

with open('requirements.txt') as f:
    my_requirements = f.read().splitlines()


def main():
    setup(
        name="otto-benchmark",
        version="1.0",
        author="A. Loisy",
        author_email="aurore.loisy@gmail.com",
        description="a fork of OTTO, "
                    "used for benchmarking solvers on the olfactory search POMDP",
        long_description=my_long_description,
        long_description_content_type="text/markdown",
        url='https://github.com/auroreloisy/otto-benchmark',
        license=my_license,
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.8",
        install_requires=my_requirements,
    )


if __name__ == '__main__':
    main()
