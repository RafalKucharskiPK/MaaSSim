from distutils.core import setup

setup(
    name='MaaSSim',
    packages=['MaaSSim'],
    version='0.9.5',
    license='MIT',
    description="agent-based two-sided mobility platform simulator",
    author='Rafal Kucharski',
    author_email='rkucharski@pk.edu.pl',
    url='https://github.com/RafalKucharskiPK/MaaSSim',
    download_url='https://github.com/RafalKucharskiPK/MaaSSim/archive/0.9.2.tar.gz',
    keywords=['MaaS', 'Uber', 'TNC', 'agent based modelling', 'two-sided mobility platforms'],
    include_package_data=True,
    install_requires=['simpy>=3.0.11',
                      'networkx>=2.4',
                      'numpy>=1.18.5',
                      'pandas>=1.0.5',
                      'dotmap>=1.2.20',
                      'osmnx>=0.15.0',
                      'scipy>=1.4.1',
                      'seaborn>=0.10.1',
                      'matplotlib>=3.2.2',
                      'exmas==0.9.99'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',  # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',  # Again, pick a license
        'Programming Language :: Python :: 3.7',
    ],
    data_files=[("data", ["MaaSSim/data/config.json",
                          "MaaSSim/data/Nootdorp.csv",
                          "MaaSSim/data/Nootdorp.graphml"])]
)
