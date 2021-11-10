from setuptools import setup, find_packages

setup(
    name="pydaptivefiltering",
    packages=find_packages(
        include=["pydaptivefiltering", "pydaptivefiltering.*"]),
    version='0.3',
    description=" Python 3.8+ Implementation of the AdaptiveFiltering Toolbox.",
    author="Bruno Lima Netto",
    author_email="brunolimanetto@gmail.com",
    url="https://github.com/BruninLima",
    download_url="https://github.com/BruninLima/PydaptiveFiltering/archive/refs/tags/v0.3.tar.gz",
    keywords=["Adaptive", "Filtering", "Digital", "Signal", "Processing"],
    install_requires=[
        'numpy',

    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        'Programming Language :: Python :: 3'
    ]

)
