'''
Author: Owen A. Johnson, strongly inspired by Greg Hellbourg's code of the same name. 
Code Purpose: Reads nenufar .raw file and generate .fil files. 
Arguments: Directory containing the raw files and frequency resolution
Date: 21-03-2024
'''

import logging
import os
import argparse
import glob
import numpy as np 
import cupy as cp
import time 
import datetime
from pathlib import Path
from astropy.time import Time

# --- Configuring logger ---
logger_name = "raw2fil_v2"
logger = logging.getLogger(logger_name)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# --- Functions ---
def log_print(*args, **kwargs):
    logger.info(*args, **kwargs)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Reads nenufar .raw file and generates .fil files")
    parser.add_argument("directory", nargs='?', type=str, help="Directory containing the .raw files")
    parser.add_argument("-f", "--freqres", type=float, help="Frequency resolution in Hz")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    return parser.parse_args()

header_keyword_types = {
    b'telescope_id' : b'<l',
    b'machine_id'   : b'<l',
    b'data_type'    : b'<l',
    b'barycentric'  : b'<l',
    b'pulsarcentric': b'<l',
    b'nbits'        : b'<l',
    b'nsamples'     : b'<l',
    b'nchans'       : b'<l',
    b'nifs'         : b'<l',
    b'nbeams'       : b'<l',
    b'ibeam'        : b'<l',
    b'rawdatafile'  : b'str',
    b'source_name'  : b'str',
    b'az_start'     : b'<d',
    b'za_start'     : b'<d',
    b'tstart'       : b'<d',
    b'tsamp'        : b'<d',
    b'fch1'         : b'<d',
    b'foff'         : b'<d',
    b'refdm'        : b'<d',
    b'period'       : b'<d',
    b'src_raj'      : b'<d',
    b'src_dej'      : b'<d'
    }

def to_sigproc_keyword(keyword, value=None):
    """ 
    Purpose: Generate a serialized string for a sigproc keyword:value pair
    If value=None, just the keyword will be written with no payload.
    Data type is inferred by keyword name (via a lookup table)
    Args:
        keyword (str): Keyword to write
        value (None, float, str, double or angle): value to write to file
    Returns:
        value_str (str): serialized string to write to file.
    """

    keyword = bytes(keyword)

    if value is None:
        return np.int32(len(keyword)).tobytes() + keyword
    else:
        dtype = header_keyword_types[keyword]

        dtype_to_type = {b'<l'  : np.int32,
                         b'str' : str,
                         b'<d'  : np.float64}

        value_dtype = dtype_to_type[dtype]

        if value_dtype is str:
            return np.int32(len(keyword)).tobytes() + keyword + np.int32(len(value)).tobytes() + value
        else:
            return np.int32(len(keyword)).tobytes() + keyword + value_dtype(value).tobytes()
        
def get_file_size(file_path):
    # Check if the file exists
    if os.path.exists(file_path):
        # Get the size of the file in bytes
        file_size = os.path.getsize(file_path)
        return file_size
    else:
        return -1


# --- Main ---
def main():
    # - Printing argument information - 
    log_print("Reading raw files and generating fil files")
    args = parse_arguments()
    verbose = args.verbose
    raw_files = sorted(glob.glob(f"{args.directory}/*.raw"))
    parset_files = sorted(glob.glob(f"{args.directory}/*.parset")); parset_nfiles = len(parset_files)

    log_print(f"Found {len(raw_files)} raw files")
    log_print(f"Found {len(parset_files)} parset files")
    log_print(f"Frequency resolution: {args.freqres} Hz")

    if verbose:
        print("Verbose mode activated")
        print(f"Found {raw_files}")
        print(f"Found {parset_files}")

    if len(raw_files) == 0:
        print("No raw files found")
        raise SystemExit

    if parset_nfiles == 0: 
        print("No parset files found")
        raise SystemExit

    # - Constants - 
    fftlen = 1 # FFT length
    npols = 4 # Number of polarizations

    # - Header Structure -
    dt_header = np.dtype([('nchans', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
    ('nsamps', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
    ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
    ])

    dt_lane_beam_chan = np.dtype([('lane', 'int32'), # LANE number
    ('beam', 'int32'), # BEAM number
    ('chan', 'int32'), # CHANNEL number
    ])

    # - Reading raw files -
    mem_pool = cp.get_default_memory_pool() # Gets current memory pool

    for raw_file in raw_files:
        with open(raw_file, 'rb') as f_raw: # read binary file
            header = np.frombuffer(f_raw.read(dt_header.itemsize), count=1, dtype=dt_header)[0] # read header
            nchans = header['nchans']; nsamps = header['nsamps']; bytespersample = header['bytespersample']
            total_bytes = nchans * nsamps * bytespersample # total bytes in file

            if verbose:
                print(f"\n=== Verbose .Raw Information == ")
                print(f"Shape: {header}")
                print(f"Number of channels: {nchans}")
                print(f"Number of samples: {nsamps}")
                print(f"Bytes per sample: {bytespersample}")
                print(f"Total bytes in file: {total_bytes}")

            # - Header Structure - 
            dt_header = np.dtype([('nchans', 'int32'), 
                                 ('nsamps', 'int32'), 
                                 ('bytespersample', 'int32'), 
                                 ('lbc_allocation', dt_lane_beam_chan, (nchans,))]) #lane-beam-channel allocation
            
            # - Data Structure - 
            dt_block = np.dtype([('eisb', 'uint64'),
                                ('tsb', 'uint64'), # Unix time of subband 
                                ('bsnb', 'uint64'), # Sample offset from current second 
                                ('data', 'int8', (total_bytes))])   # data block structure
                                    
            with open(raw_file,'rb') as fd_raw:
                header = np.frombuffer(fd_raw.read(dt_header.itemsize), # reading header bytes 
                count=1,
                dtype=dt_header,
                )[0] #  Accesses the first element of the array created by np.frombuffer as count = 1
            log_print(f"Reading {raw_file} header...")
            
            data = np.memmap(raw_file, # memory map the file
                    dtype=dt_block, # data type
                    mode='r', # read only mode
                    offset=dt_header.itemsize, # offset to skip the header
                    )
            log_print(f"Reading {raw_file} data block...")

            nfft = len(data[0]['data']) // nchans // fftlen // npols # number of ffts
            nBlocks = len(data) # number of blocks
            nRes = (2**round(np.log2(200.1e6/1024./args.freqres)))
            NumBck = int(np.ceil(nRes / nfft))

            if verbose:
                print(f"\n=== Verbose Resolution Information == ")
                print(f"Number of FFTs: {nfft}")
                print(f"Number of Resolution Channels: {nRes}")
                print(f"Number of Blocks: {NumBck}\n")


            # - .fil file names  -
            filnames = [] # list to store the output .fil names
            for nBeam in range(header[0]): # loop over beams
                fname = os.path.join(os.path.dirname(raw_file),args.directory,'lane'+str(header[3][nBeam][0]).zfill(2)+'_beam'+str(header[3][nBeam][1]).zfill(3)+'_chan'+str(header[3][nBeam][2]).zfill(3)+'.fil'); # .fil names
                # example output lane00_beam000_chan342.fil
                open_file = open(fname, 'wb') # open file in write binary mode
                filnames.append(open_file)

            # - Processing the .raw files to filterbanks -
            dsetft = cp.ndarray((nRes,1,nchans,4),dtype=cp.csingle) # 4D array 

            log_print(f"Processing {raw_file}...")
            log_print(f"Channelizing Resolution: {(200.1e6/1024./nRes)} Hz")
            log_print(f"Time Resolution: {(1./(200.*1e6/1024.) * NumBck*nfft)} seconds")

            start_time = time.time()

            nIter = int(np.floor(nBlocks/NumBck))
            for nBlock in range(nIter): 
                
                log_print(f"Processing block {nBlock+1} of {nIter}")
                
                # - Reading, reshaping and concatenating data from .raw files -
                data_set = data[nBlock*NumBck:(nBlock+1)*NumBck]['data']
                data_set = data_set.reshape(NumBck, nfft, fftlen, nchans, npols)
                data_set = np.concatenate(data_set, axis = 0)
                nonzeroindexes = np.where(np.sum(np.sum(data_set,axis=0),axis=-1)[0,:]!=0)[0] # find non-zero channels and sum over the 0 and -1 axis 
                data_set = data_set[:,:,nonzeroindexes,:] # remove empty channels

                dsetft = cp.fft.fft(cp.asarray(data_set), n=nRes, axis=0) # 1D FFT along the 0 axis 

                # - Power Spectrum Calculation -
                pwrspec = cp.power(dsetft[:,:,:,0].real - dsetft[:,:,:,1].imag, 2)+\
                        cp.power(dsetft[:,:,:,0].imag + dsetft[:,:,:,1].real, 2)+\
                        cp.power(dsetft[:,:,:,2].real - dsetft[:,:,:,3].imag, 2)+\
                        cp.power(dsetft[:,:,:,2].imag + dsetft[:,:,:,3].real, 2)

                pwrspec = cp.fft.fftshift(pwrspec, axes=0) # shift the zero frequency to the center of the spectrum

                for nBeam in range(header[0]):
                    filnames[nBeam].write(cp.asnumpy(pwrspec[:,0,nBeam]).astype(np.uint32)) # write the power spectrum to the .fil file for each beam
                
                mem_pool.free_all_blocks() # free all memory blocks

            for nBeam in range(header[0]):
                filnames[nBeam].close()

            end_time = time.time()
            log_print(f"Processing time: {(end_time-start_time)/60} mins")

            # - Creating .sigproc header - 
            filfiles = glob.glob(args.directory + '*.fil') # list of .fil files that have been created. 
            if verbose: 
                print(f"\nFound {len(filfiles)} .fil files")

            parset_file = open(parset_files[0], "r") # open the parset file
            parset_body = parset_file.readlines()

            for k in range(len(parset_body)):
                if parset_body[k].find('.nrBeams=') != -1: # if the string '.nrBeams=' is found
                    idx = parset_body[k].find('.nrBeams=') # find the index of the string '.nrBeams='
                    nBeams = int(parset_body[k][idx+9:-1]) # number of beams

            log_print(f"Number of beams found in parset: {nBeams}")

            for nSource in range(nBeams): 
                for k in range(len(parset_body)):
                    # - Extracting information from the parset file using find - 
                    if parset_body[k].find('Beam[' + str(nSource) + '].target=') != -1: 
                        idx = parset_body[k].find('target=');
                        targetname = parset_body[k][idx+7:-1];
                    if parset_body[k].find('Beam[' + str(nSource) + '].subbandList=[') != -1:
                        idx = parset_body[k].find('List=[');
                        chanlow = int(parset_body[k][idx+6:parset_body[k].find('..')]);
                        chanhi = int(parset_body[k][parset_body[k].find('..')+2:-2]);
                    if parset_body[k].find('Beam[' + str(nSource) + '].angle1=') != -1:
                        idx = parset_body[k].find('angle1=');
                        ang1 = float(parset_body[k][idx+7:-1]);
                    if parset_body[k].find('Beam[' + str(nSource) + '].angle2=') != -1:
                        idx = parset_body[k].find('angle2=');
                        ang2 = float(parset_body[k][idx+7:-1]);
                    if parset_body[k].find('Beam[' + str(nSource) + '].startTime=') != -1:
                        idx = parset_body[k].find('startTime=');
                        timeobsstr = parset_body[k][idx+10:-1];
                        timeobs = datetime.datetime.strptime(timeobsstr,'%Y-%m-%dT%H:%M:%SZ')
            
            beam_filenames = []; fsizes = []; misschan = []
            
            for k in range(chanlow,chanhi+1):
                beam_filenames.append(glob.glob(args.directory + '*_beam' + str(nSource).zfill(3)+'_chan'+str(k).zfill(3)+'*.fil'))
                if beam_filenames[-1] == []: # in case channels are missing
                    misschan.append(k)
                    fsizes.append(0)
                else:
                    fsizes.append(get_file_size(beam_filenames[-1][0]))

            log_print(f"Target: {targetname}")
            log_print(f"Channel Low: {chanlow}")
            log_print(f"Channel High: {chanhi}")
            log_print(f"Angle 1: {ang1}")
            log_print(f"Angle 2: {ang2}")
            log_print(f"Time Observed: {timeobs}")
            log_print(f"Splicing {len(beam_filenames)} files")
            if len(misschan) == 0:
                log_print("All channels found")
            else:
                log_print(f"Missing channels: {misschan}")

            channumber = range(chanlow, chanhi + 1)

            # - preparing the header - 
            fil_header = {b'telescope_id': b'66',    # NenuFAR
                b'nbits': str(32).encode(),       # TBD
                b'source_name': targetname.encode(),   # read in parset AnaBeam[0].directionType
                b'data_type': b'1',       #  TODO: look into that
                b'nchans': str(nRes * len(channumber)).encode(), # 2**17 x number of channels
                b'machine_id': b'99', # ??
                b'tsamp': str(1./(200.*1e6/1024.) * NumBck*nfft).encode(),
                b'foff': str(200./1024./nRes).encode(),    # 200./1024./2**18
                b'src_raj': str(ang1).encode(),
                b'src_dej': str(ang2).encode(),
                b'tstart': str(Time(timeobsstr).mjd).encode(),
                b'nbeams': b'1',
                b'fch1': str(chanlow*200.0/1024 + 200./1024./2).encode(),
                b'nifs': str(len(channumber)).encode()}
            
            header_string = b''
            header_string += to_sigproc_keyword(b'HEADER_START')

            for keyword in fil_header.keys():
                if keyword not in header_keyword_types.keys():
                    pass
                else:
                    header_string += to_sigproc_keyword(keyword, fil_header[keyword])
        
            header_string += to_sigproc_keyword(b'HEADER_END')

            # - Splicing the .fil files and writing final filterbank - 
            filsavename = (args.directory + str(timeobs.year) + str(timeobs.month) + str(timeobs.day) + str(timeobs.hour) + str(timeobs.minute) + str(timeobs.second) + '_' + targetname.strip('"') + '_' + str(np.round(200.1e6/1024./nRes, 2)) + 'Hz.fil')

            log_print(f"Saving to {filsavename}")

            output_file = open(filsavename, 'wb')
            output_file.write(header_string)

            input_files = []
            for nBeam in range(len(beam_filenames)):
                if beam_filenames[nBeam] == []:
                    input_files.append([])
                else:
                    f = open(beam_filenames[nBeam][0], 'rb') 
                    input_files.append(f)

            nIdx = 0
            smallfile = np.argmin(np.array(fsizes)[np.nonzero(fsizes)[0]])
            numfiles = len(input_files)

            while input_files[smallfile].read(1):
                if verbose:
                    print('Writing spectrum #'+str(nIdx))
                for k in range(numfiles):
                    if channumber[k] in misschan:
                        output_file.write(np.zeros((nRes)).astype(np.uint32))
                    else:
                        input_files[k].seek(int(4*nRes*nIdx))
                        data_to_write = input_files[k].read(int(4*nRes))
                        output_file.write(data_to_write)
                        if verbose: 
                            log_print(f"Writing {len(data_to_write)} bytes")
                nIdx += 1

            log_print(f"Output filterbank size {output_file.tell()/(1024**3)} Gb")
            output_file.close()    
                
            for nBeam in range(numfiles):
                input_files[nBeam].close();


if __name__ == "__main__":
    main()
