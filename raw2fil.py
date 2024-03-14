''' 
Author: Greg Hellbourg
Purpose: reads nenufar raw file and generate fil files, python raw2fil.py /datax2/devfil/ 1.0
Arguments : directory containing the raw files and frequency resolution
'''


import argparse
import numpy as np
import cupy as cp
import time
from pathlib import Path
import sys
import os
from astropy.time import Time
import glob
import matplotlib.pyplot as plt
import datetime
import logging

workdirec =  sys.argv[1];
#workdirec = '/datax2/devfil/';
resval =  float(sys.argv[2]);   # resolution in Hz

logging.basicConfig(level=logging.INFO, format='%(message)s');
logger = logging.getLogger();
logger.addHandler(logging.FileHandler(os.path.join(workdirec,'logreduce.log'), 'a'));
print = logger.info;

directory = 'beam_data';
path = os.path.join(workdirec,directory);
if not os.path.exists(path):
    os.mkdir(path);

fftlen = 1;
nof_polcpx = 4;

fnames_raw = sorted(list(glob.glob(workdirec+'*.raw')));
if fnames_raw == []:
    print("No raw file in working directory. Exiting.");
    sys.exit();

fnames_parset = sorted(list(glob.glob(workdirec+'*.parset')));
nFiles = len(fnames_raw);

dt_header = np.dtype([('nobpb', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
    ('nb_samples', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
    ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
    ]);
    
dt_lane_beam_chan = np.dtype([('lane', 'int32'),('beam', 'int32'), ('chan', 'int32')])    # lane number, beam number, and channel number

mempool = cp.get_default_memory_pool();

#########################
for fname in fnames_raw:
#fname = fnames_raw[0];
#########################

    with open(fname,'rb') as fd_raw:
        header = np.frombuffer(fd_raw.read(dt_header.itemsize),
                               count=1,
                               dtype=dt_header,
                               )[0];

    nobpb = header['nobpb']
    nb_samples = header['nb_samples']
    bytespersample = header['bytespersample']

    bytes_in = bytespersample * nb_samples * nobpb

    dt_block = np.dtype([('eisb', 'uint64'),
       ('tsb', 'uint64'),
       ('bsnb', 'uint64'),
       ('data', 'int8', (bytes_in,)),
       ])   # data block structure

    dt_header = np.dtype([('nobpb', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
      ('nb_samples', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
      ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
      ('lbc_alloc', dt_lane_beam_chan, (nobpb,)),
      ])    # header structure
        
    with open(fname,'rb') as fd_raw:
        header = np.frombuffer(fd_raw.read(dt_header.itemsize),
           count=1,
           dtype=dt_header,
           )[0]

    data = np.memmap(fname,
     dtype=dt_block,
     mode='r',
     offset=dt_header.itemsize,
     )

    nfft = len(data[0]['data']) // nobpb // fftlen // nof_polcpx;
    # nfft = 87296
    nBlocks = len(data);
    nRes = int(2**np.floor(np.log2(200.1e6/1024./resval)));    ## resolution about 1 Hz
    NumBck = int(np.ceil(nRes / nfft)); ## reads multiple blocks to reach the desired resolution

    outfiles = [];
    for nBeam in range(header[0]):
        Filfname = os.path.join(os.path.dirname(fname),directory,'lane'+str(header[3][nBeam][0]).zfill(2)+'_beam'+str(header[3][nBeam][1]).zfill(3)+'_chan'+str(header[3][nBeam][2]).zfill(3)+'.fil'); # FIL file name
        f = open(Filfname, 'wb');
        outfiles.append(f);

    #spec = cp.zeros((nRes,1,nobpb));
    #specwrite = np.zeros((nRes,1,nobpb),dtype=np.uint8);
    dsetft = cp.ndarray((nRes,1,nobpb,4),dtype=cp.csingle);

    print('processing file : ' + fname);
    print('channelizing at ' + str(200.*1e6/1024./nRes) + ' Hz resolution.');
    print(str(nBlocks) + ' blocks to process.');
    start = time.time();

    nIter = int(np.floor(nBlocks/NumBck));
    for nbck in range(nIter):
        
#        print('processing block '+str(nbck+1)+' / '+str(nIter));
        dset = data[nbck*NumBck:(nbck+1)*NumBck]['data'];
        dset.shape = (NumBck,nfft, fftlen, nobpb, nof_polcpx);
        dset = np.concatenate((dset),axis=0);
        nonzeroidx = np.where(np.sum(np.sum(dset,axis=0),axis=-1)[0,:]!=0)[0];
        dset = dset[:,:,nonzeroidx,:];
        dsetft = cp.fft.fft(cp.asarray(dset),n=nRes,axis=0);

        spec = cp.power(dsetft[:,:,:,0].real - dsetft[:,:,:,1].imag,2)+\
            cp.power(dsetft[:,:,:,0].imag + dsetft[:,:,:,1].real,2)+\
            cp.power(dsetft[:,:,:,2].real - dsetft[:,:,:,3].imag,2)+\
            cp.power(dsetft[:,:,:,2].imag + dsetft[:,:,:,3].real,2);
            
        spec = cp.fft.fftshift(spec,axes=0);
        
        for nBeam in range(spec.shape[-1]):
            outfiles[nonzeroidx[nBeam]].write(cp.asnumpy(spec[:,0,nBeam]).astype(np.uint32));
        
        mempool.free_all_blocks();
                
    for nBeam in range(header[0]):
        outfiles[nBeam].close();
        
    stop = time.time();
    print(str(stop-start) + 's elapsed');



## delete raw files
for fname in fnames_raw:
    os.remove(fname);


## splice files together


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
    b'src_dej'      : b'<d',
#    b'src_raj'      : b'angle',
#    b'src_dej'      : b'angle',
    }
    
def to_sigproc_keyword(keyword, value=None):
    """ Generate a serialized string for a sigproc keyword:value pair
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
        return np.int32(len(keyword)).tostring() + keyword
    else:
        dtype = header_keyword_types[keyword]

        dtype_to_type = {b'<l'  : np.int32,
                         b'str' : str,
                         b'<d'  : np.float64}#,
#                         b'angle' : to_sigproc_angle}

        value_dtype = dtype_to_type[dtype]

        if value_dtype is str:
            return np.int32(len(keyword)).tostring() + keyword + np.int32(len(value)).tostring() + value
        else:
            return np.int32(len(keyword)).tostring() + keyword + value_dtype(value).tostring()




filfiles = glob.glob(os.path.join(workdirec,directory,'lane*.fil'));

fnames_parset = sorted(list(glob.glob(workdirec+'*.parset')));
if fnames_parset == []:
    print("No parset file in working directory. Exiting.");
    sys.exit();
if len(fnames_parset) > 1:
    print("Directory contains more than 1 parset file. Exiting.");
    sys.exit();

f = open (fnames_parset[0], "r");
data = f.readlines();
for k in range(len(data)):
    if data[k].find('.nrBeams=') != -1:
        idx = data[k].find('.nrBeams=');
        nBeams = int(data[k][idx+9:-1]);
        
print('Found ' + str(nBeams) + ' beams.');
        
for nSource in range(nBeams):
    for k in range(len(data)):
        if data[k].find('Beam[' + str(nSource) + '].target=') != -1:
            idx = data[k].find('target=');
            targetname = data[k][idx+7:-1];
        if data[k].find('Beam[' + str(nSource) + '].subbandList=[') != -1:
            idx = data[k].find('List=[');
            chanlow = int(data[k][idx+6:data[k].find('..')]);
            chanhi = int(data[k][data[k].find('..')+2:-2]);
        if data[k].find('Beam[' + str(nSource) + '].angle1=') != -1:
            idx = data[k].find('angle1=');
            ang1 = float(data[k][idx+7:-1]);
        if data[k].find('Beam[' + str(nSource) + '].angle2=') != -1:
            idx = data[k].find('angle2=');
            ang2 = float(data[k][idx+7:-1]);
        if data[k].find('Beam[' + str(nSource) + '].startTime=') != -1:
            idx = data[k].find('startTime=');
            timeobsstr = data[k][idx+10:-1];
            timeobs = datetime.datetime.strptime(timeobsstr,'%Y-%m-%dT%H:%M:%SZ')

    ## beamfname = glob.glob(os.path.join(path,'*_beam'+str(nSource).zfill(3)+'*'));
    beamfname = [];
    fsizes = [];
    misschan = [];
    for k in range(chanlow,chanhi+1):
        beamfname.append(glob.glob(os.path.join(path,'*_beam'+str(nSource).zfill(3)+'_chan'+str(k).zfill(3)+'*')));
        if beamfname[-1] == []: # in case channels are missing
            misschan.append(k);
            fsizes.append(0);
        else:
            fsizes.append(Path(beamfname[-1][0]).stat().st_size);
        
    print('target = ' + targetname);
    print('channel low = ' + str(chanlow));
    print('channel high = ' + str(chanhi));
    print('angle 1 = ' + str(ang1));
    print('angle 2 = ' + str(ang2));
    print('observed at : ' + timeobsstr);
    print('splicing '+ str(len(beamfname)) + ' files');
    if len(misschan) == 0:
        print('no missing channel.');
    else:
        print('missing channels:');
        for mc in misschan:
            print(str(mc));
    print('');
    
    channum = range(chanlow,chanhi+1);
    
    # prepare header
    f = {b'telescope_id': b'66',    # NenuFAR
      b'nbits': str(32).encode(),       # TBD
      b'source_name': targetname.encode(),   # read in parset AnaBeam[0].directionType
      b'data_type': b'1',       # look into that
      b'nchans': str(nRes * len(channum)).encode(), # 2**17 x number of channels
      b'machine_id': b'99', # ??
      b'tsamp': str(1./(200.*1e6/1024.) * NumBck*nfft).encode(),
      b'foff': str(200./1024./nRes).encode(),    # 200./1024./2**18
      b'src_raj': str(ang1).encode(),
      b'src_dej': str(ang2).encode(),
      b'tstart': str(Time(timeobsstr).mjd).encode(),
      b'nbeams': b'1',
      b'fch1': str(chanlow*200.0/1024 + 200./1024./2).encode(),
      b'nifs': str(len(channum)).encode()}

    header_string = b'';
    header_string += to_sigproc_keyword(b'HEADER_START');
    
    for keyword in f.keys():
        if keyword not in header_keyword_types.keys():
            pass;
        else:
            header_string += to_sigproc_keyword(keyword, f[keyword]);
        
    header_string += to_sigproc_keyword(b'HEADER_END');
    
    
    
    Filfname = os.path.join(workdirec, directory, str(timeobs.year) + str(timeobs.month) + str(timeobs.day) + str(timeobs.hour) + str(timeobs.minute) + str(timeobs.second) + '_' + targetname.strip('"') + '.fil'); # FIL file name
    fout = open(Filfname, 'wb');
    fout.write(header_string);
    print('writing to ' + Filfname);
    
    infiles = [];
    for nBeam in range(len(beamfname)):
        if beamfname[nBeam] == []:
            infiles.append([]);
        else:
            f = open(beamfname[nBeam][0], 'rb');
            infiles.append(f);
    
    nIdx = 0;
    smallfile = np.argmin(np.array(fsizes)[np.nonzero(fsizes)[0]]);
    numfiles = len(infiles);
    while infiles[smallfile].read(1):
#        print('writing spectrum #'+str(nIdx));
        for k in range(numfiles):
            if channum[k] in misschan:
                fout.write(np.zeros((nRes)).astype(np.uint32));
            else:
                infiles[k].seek(int(4*nRes*nIdx))
                fout.write(infiles[k].read(int(4*nRes)));
        nIdx += 1;
        
    for nBeam in range(numfiles):
        infiles[nBeam].close();
        
## remove non-spliced fil files
for fname in filfiles:
    os.remove(fname);
