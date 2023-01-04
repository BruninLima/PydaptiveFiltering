from setuptools import setup, find_packages

setup(
    name="pydaptivefiltering",
    packages=find_packages(
        include=["pydaptivefiltering", "pydaptivefiltering.*"]),
    version='0.5',
    description="A modern Python implementation of the AdaptiveFiltering toolbox.",
    author="Bruno Lima Netto",
    author_email="brunolimanetto@gmail.com",
    url="https://github.com/BruninLima",
    download_url="https://github.com/BruninLima/PydaptiveFiltering/archive/refs/tags/v0.3.tar.gz",
    keywords=["Adaptive", "Filtering", "Digital", "Signal", "Processing"],
    install_requires=[
        'numpy',
        'time',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3'
    ]

)
