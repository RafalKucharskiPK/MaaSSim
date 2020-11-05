from distutils.core import setup

setup(
    name='MaaSSim',
    packages=['MaaSSim'],  # Chose the same as "name"
    version='0.9',  # Start with a small number and increase it with every change you make
    license='MIT',  # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="agent-based two-sided mobility platform simulator",
    author='Rafal Kucharski',  # Type in your name
    author_email='rkucharski@pk.edu.pl',  # Type in your E-Mail
    url='https://github.com/RafalKucharskiPK/MaaSSim',  # Provide either the link to your github or to your website
    download_url='https://github.com/RafalKucharskiPK/MaaSSim/archive/1.0.tar.gz',  # I explain this later on
    keywords=['MaaS', 'Uber', 'TNC', 'agent based modelling', 'two-sided mobility platforms'],  # Keywords that define your package best
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
)
