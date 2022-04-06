# SMC EFR

This is my dataset for the Smoky Mountain Conference Data Challenge ([https://smc-datachallenge.ornl.gov/](https://smc-datachallenge.ornl.gov/)). 


See the file `./dl-smcefr.py` for technical specs. The basic idea is to download EFR (Earth Full Resolution) from the Sentinel-3 satellite (via [Copernicus](https://scihub.copernicus.eu/)), and reduce the multispectral data into standard RGB image formats, cropped to (1024, 1024, 3) and stored as a directory of PNG files (`./data`)

## Download Data

To download the dataset, run:

```shell
$ pip3 install pillow numpy netCDF4
$ python3 ./dl-smcefr.py
```
