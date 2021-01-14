from distutils.core import setup

setup(
    name="Equivariant Attention",
    version="0.0.1",
    description="Regular representation based equivariant attention layers",
    author="Michael Hutchinson, Charline Le Lan, Sheh Zaidi",
    packages=["eqv_transformer"],
    python_requires=">=3.7",
    install_requires=[
        "torch==1.6.0",
        "torchvision==0.7.0",
        "tensorboard",
        "numpy",
        "attrdict",
        "simplejson",
        "future",
        "deepdish",
        "tqdm",
        "einops",
        "forge @ git+https://github.com/akosiorek/forge",
        "lie-conv @ git+https://github.com/MJHutchinson/LieConv",
    ],
    long_description=open("README.md").read(),
)
