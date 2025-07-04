from setuptools import find_packages,setup

def get_requirements(file_path:str)->list[str]:
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name= 'ML_Project',
    version='1.0',
    author='Samarth Srivastava',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)