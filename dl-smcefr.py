#!python3
""" dl-smcefr.py - download the Sentinel-3 data from the Copernicus Open Data Hub

This does preliminary filtering, and production of RGB images in `./data`, named according
  to Sentinel-3 metadata. Or, you can just view it as a big batch of images

Requires that you have an account:

  * https://scihub.copernicus.eu/dhus/#/user-profile

Run `pip3 install requirements.txt`

@author: Cade Brown <me@cade.site>
"""

from concurrent.futures import ThreadPoolExecutor
import json
import multiprocessing
from multiprocessing.connection import wait
import sys
import io
import os
from getpass import getpass
import time
from threading import Lock
import requests
import tempfile

# general numerics
import numpy as np

# for outputting images
from PIL import Image

# for reading .nc file
import netCDF4

# API root
API_ROOT = 'https://scihub.copernicus.eu/dhus/odata/v1'

# username/password
API_USER = sys.argv[1]
#API_PASS = input(f"password for '{API_USER}': ")
API_PASS = getpass(f"password for '{API_USER}': ")

# for API/internet access
#API_LOCK = Lock()

# HTTP session
sess = requests.Session()
sess.auth = (API_USER, API_PASS)

### UTILITIES ###

def GET(url):
    res = sess.get(url)
    if not res.ok:
        print (f'FAILED: GET {url}')
        print('  dump: ', res.content)
        raise Exception(repr(res))

    # maximum of 2 req/sec
    time.sleep(0.5)

    return res


def NC(data):
    # convert binary data to netCDF4 dataset

    # we have to use a temporary file
    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(data)
        return netCDF4.Dataset(tmp.name)


### ROUTINES ###

def do_efr(prodid):
    # get EFR data from the product ID
    print (f'ID: {prodid}')

    prodname = GET(f"{API_ROOT}/Products('{prodid}')/Name/$value").text
    print (f'  name: {prodname}')
    output = f"./data/{prodname}.png"

    if os.path.exists(output):
        print ('  ALREADY EXISTS')

    # download neccessary data
    #nc14 = NC(GET(f"{API_ROOT}/Products('{prodid}')/Nodes('{prodname}.SEN3')/Nodes('Oa14_radiance.nc')/$value").content)

    nc8 = NC(GET(f"{API_ROOT}/Products('{prodid}')/Nodes('{prodname}.SEN3')/Nodes('Oa08_radiance.nc')/$value").content)
    nc6 = NC(GET(f"{API_ROOT}/Products('{prodid}')/Nodes('{prodname}.SEN3')/Nodes('Oa06_radiance.nc')/$value").content)
    nc4 = NC(GET(f"{API_ROOT}/Products('{prodid}')/Nodes('{prodname}.SEN3')/Nodes('Oa04_radiance.nc')/$value").content)


    "https://scihub.copernicus.eu/dhus/odata/v1/Products('83f3069f-4b25-49f0-8137-511a7453e8f6')/Nodes('S3A_OL_1_EFR____20211230T134025_20211230T134325_20211231T184123_0179_080_181_2700_LN1_O_NT_002.SEN3')/Nodes('Oa08_radiance.nc')"


    """
    var band1 = brightness * (stretch(B09, 0, 0.25)-0.1*stretch(B14, 0, 0.1));
    var band2 = brightness * (1.1*stretch(B06, 0, 0.25)-0.1* stretch(B14, 0, 0.1));
    var band3 = brightness *  (stretch(B04, 0, 0.25)-0.1*stretch(B14, 0, 0.1)+.01*stretch(index, 0.5, 1));

    """
    def unpack(ncv):
        # unpack into normal float array
        return np.array(ncv[:], dtype=np.float32) * ncv.scale_factor + ncv.add_offset

    # color channels
    R = unpack(nc8.variables['Oa08_radiance'])
    G = unpack(nc6.variables['Oa06_radiance'])
    B = unpack(nc4.variables['Oa04_radiance'])

    # convert to single RGB image
    pix = np.dstack((R, G, B))

    sparsity = 1.0 - (np.count_nonzero(nc8.variables['Oa08_radiance']) / float(pix.size))
    print (f"  sparsity: {sparsity}")

    if sparsity > 0.7:
        print (f"  NO OUTPUT (NOT INTERESTING)")
        return

    # just crop to center
    if pix.shape[0] >= 1024 and pix.shape[1] >= 1024:
        l0 = (pix.shape[0] - 1024) // 2
        l1 = (pix.shape[1] - 1024) // 2
        # perform crop
        pix = pix[l0:l0+1024, l1:l1+1024, :]
    else:
        print (f"  NO OUTPUT (CLIPPED)")
        return

    # otherwise, output the image
    Image.fromarray((np.clip(pix, 0, 1) * 255).astype(np.uint8)).save(output)
    print (f"  output: {output}")



def mainloop(query, numres=1000):
    # search for data products

    def perid(id):
        # utility function
        try:
            do_efr(id)
        except Exception as e:
            print ("EXCEPTION: ", repr(e))

    with ThreadPoolExecutor(max_workers=1) as executor:
        for i in range(0, 1000, 100):
            print (f"requesting results {i} to {i+100}...")
            res = GET(f"https://scihub.copernicus.eu/dhus/search?q={query}&rows=100&start={i}&format=json").text
            
            print (f"  success: {i} to {i+100}")

            futs = []

            try:
                data = json.loads(res)['feed']['entry']

                for x in data:
                    # actually do logic here
                    futs.append(executor.submit(perid, x['id']))
                    #perid(x['id'])


                    #pool.apply
                    #fn(x['id'])
                    #yield (i + j, x['id'])
            except Exception as e:
                print ("EXCEPTION: ", repr(e))
                print ("  response: ", res)

            wait(futs, timeout=30)

#mainloop('ingestiondate:[2021-01-01T00:00:00.000Z TO 2021-06-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___')


#mainloop('ingestiondate:[2022-04-01T00:00:00.000Z TO 2022-08-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___')
#mainloop('ingestiondate:[2022-01-01T00:00:00.000Z TO 2022-04-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___')
mainloop('ingestiondate:[2022-01-01T00:00:00.000Z TO 2022-04-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___')




#mainloop('ingestiondate:[2021-06-01T00:00:00.000Z TO 2022-01-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___ AND ( footprint:"Intersects(POLYGON((-82.81167009725682 -55.31132818032862,-35.69468538674864 -55.31132818032862,-35.69468538674864 10.646602960492118,-82.81167009725682 10.646602960492118,-82.81167009725682 -55.31132818032862)))" )')

"""
Queries:

pretty good, NA region:
'ingestiondate:[2021-06-01T00:00:00.000Z TO 2022-01-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___ AND ( footprint:"Intersects(POLYGON((-126.21640752754321 13.855272652456605,-61.96597383139563 13.855272652456605,-61.96597383139563 49.48536580184964,-126.21640752754321 49.48536580184964,-126.21640752754321 13.855272652456605)))" )'

"""




"""

for id in search():

    pool.apply_async(tdo, (id,))

#pool.map(tdo, ids)

pool.close()
pool.join()

"""


#for id in ids:
#    do_efr(id)


"""
wget --no-check-certificate --user=X --password=Y "https://scihub.copernicus.eu/dhus/search?q=ingestiondate:[2022-01-01T00:00:00.000Z TO 2022-03-01T00:00:00.000Z] AND platformname:Sentinel-3 AND producttype:OL_1_EFR___&rows=100&start=0&format=json" -O -> res.csv
"""



"""



def get_lst(prodid):
    # retrieve the product name



    # 2D temperature data
    xy = ncdata.variables['LST'][:]
    assert len(xy.shape) == 2

    # scale from 230C (-46F) to 330C (+134F) to 0.2 to 1.0
    # NOTE: data that is not present is set to 0
    scl_in = (230, 330)
    scl_out = (0.2, 1.0)

    # normalized, and filled
    xy_normed = np.clip((scl_out[1] - scl_out[0]) * (xy - scl_in[0]) / (scl_out[0] - scl_in[0]) + scl_out[0], 0, 1).filled(0)

    sparsity = 1.0 - (np.count_nonzero(xy_normed) / float(xy_normed.size))
    print (f"  sparsity: {sparsity}")

    if sparsity < 0.8:
        # interesting

        # save to PNG
        output = f"{prodname}.png"
        Image.fromarray((xy_normed * 255).astype(np.uint8)).save(output)
        print (f"  output: {output}")
    else:
        # not really interesting
        print (f"  NO OUTPUT (NOT INTERESTING ENOUGH)")

    #key = 'LST'
    #key = next(iter(ncdata.variables.keys()))


    #print (sess.get(f"{API_ROOT}/Products('{prodid}')/Name/$value").text)



"""
