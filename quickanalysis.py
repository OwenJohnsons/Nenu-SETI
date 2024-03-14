'''
Author: Greg Hellbourg 
Code: Purpose, plot .raw file on NenuFAR data 
'''

import numpy as np
import matplotlib.pyplot as plt

fftlen = 1;
nof_polcpx = 4;

fname = '/datax2/ES13/2022/11/20221102_222400_20221102_223600_SETI/SETI_20221102_222436_0.raw';

dt_header = np.dtype([('nobpb', 'int32'),          # NUMBER_OF_BEAMLET_PER_BANK = number of channels
    ('nb_samples', 'int32'),     # NUMBER of SAMPLES (fftlen*nfft)
    ('bytespersample', 'int32'), # BYTES per SAMPLE (4/8 for 8/16bits data)
    ]);

dt_lane_beam_chan = np.dtype([('lane', 'int32'),('beam', 'int32'), ('chan', 'int32')])    # lane number, beam number, and channel number

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

dt_header = np.dtype([('nobpb', 'int32'),   # NUMBER_OF_BEAMLET_PER_BANK = number of channels
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

nRes = 4096;
alldat = np.zeros((nRes*nobpb,len(data)));

for k in range(len(data)):
    tmp = data[k]['data'];
    tmp.shape = (nfft, fftlen, nobpb, nof_polcpx);
    tmp = np.squeeze(tmp);
    tmp = tmp.astype('float32').view('complex64');

    spec = np.squeeze(tmp[:int(np.floor(nfft/nRes)*nRes),:,:]);
    spec = np.reshape(spec,(nRes,int(np.floor(nfft/nRes)),spec.shape[-2],spec.shape[-1]),order='F');
    spec = np.squeeze(np.sum(np.fft.fftshift(np.mean(np.abs(np.fft.fft(spec,axis=0))**2,axis=1),axes=0),axis=-1));
    
    alldat[:,k] = spec.flatten(order='F');
    print(str(k)+' / '+str(len(data)));

df = 200. / 1024.;
freqs = np.linspace(204./1024.*200.-df/2.,353./1024.*200.+df/2.,nRes*nobpb);

plt.figure();
plt.plot(freqs,10.*np.log10(np.max(alldat[:,:64],axis=-1)),label='Max');
plt.plot(freqs,10.*np.log10(np.mean(alldat[:,:64],axis=-1)),label='Mean');
plt.plot(freqs,10.*np.log10(np.median(alldat[:,:64],axis=-1)),label='Median');
plt.grid();
plt.legend();
plt.xlabel('frequency [MHz]');
plt.ylabel('power [dB]');
plt.xlim([freqs[0],freqs[-1]]);
plt.show()

plt.imshow(np.log10(alldat),aspect='auto',interpolation=None);plt.show()
