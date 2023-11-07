'''
Code Purpose: Use intial file locations and name to make a query to the Vizier and a list of target names. 
Author: Owen A. Johnson
Last Major Update: 11/07/2023
'''
#%%
import numpy as np 
from astroquery.simbad import Simbad
import pandas as pd


file_list = np.loadtxt('file-locations.txt', dtype=str)

target_name = []
for line in file_list:
    trgt = line.split('/')[-1].split('_')[1][0:-4]
    target_name = np.append(target_name, trgt)

print('Number of files on SETIBK: ', len(target_name))
# drop duplicates
target_name = np.unique(target_name)
# np.savetxt('target-names.txt', target_name, fmt='%s')
print('Number of targets with converted filterbanks on SETIBK: ', len(target_name))

loaded_targets = np.loadtxt('target-names.txt', dtype=str)

main_ids = []; ra_arr = []; dec_arr = []
for target in loaded_targets:
    result_table = Simbad.query_object(target)
    # Converting to degrees
    ra = result_table['RA'][0].split(' ')
    ra = (float(ra[0]) + float(ra[1])/60 + float(ra[2])/3600)*15
    dec = result_table['DEC'][0].split(' ')
    dec = float(dec[0]) + float(dec[1])/60 + float(dec[2])/3600
    main_id = result_table['MAIN_ID'][0]

    main_ids = np.append(main_ids, main_id); ra_arr = np.append(ra_arr, ra); dec_arr = np.append(dec_arr, dec)

# Save to .csv
df = pd.DataFrame({'Target': loaded_targets, 'Main_ID': main_ids, 'RA': ra_arr, 'DEC': dec_arr})
print(df.head())
df.to_csv('csv/target-info.csv', index=False)
# %%
