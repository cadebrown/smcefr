# SMC EFR

This is my dataset for the Smoky Mountain Conference Data Challenge ([https://smc-datachallenge.ornl.gov/](https://smc-datachallenge.ornl.gov/)). 

You can see the source document at `./smcefr.md` (which contains markdown sources)

To convert to another format, use pandoc (`apt install pandoc`), like so:

```shell
$ pandoc -s smcefr.md -o index.html
```


See the file `./dl-smcefr.py` for technical specs. The basic idea is to download EFR (Earth Full Resolution) from the Sentinel-3 satellite (via [Copernicus](https://scihub.copernicus.eu/)), and reduce the multispectral data into standard RGB image formats, cropped to (1024, 1024, 3) and stored as a directory of PNG files (`./data`)

## Download Data

Navigate to the [GitHub releases](https://github.com/cadebrown/smcefr/releases), and download `smcefr-mini.tar.gz`

Then, you can expand the tar file, and have a directory of the PNG image dataset

## Generate Data

To generate the dataset by yourself (WARNING: uses lots of network/disk access), run:

```shell
$ pip3 install pillow numpy netCDF4
$ python3 ./dl-smcefr.py
```
