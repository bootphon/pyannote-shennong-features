# pyannote-shennong-features
A feature extractor for Pyannote that is based on the Shennong library

## Installation

This library depends on Shennong, which is available on conda. 
Thus, make sure you're in a conda environment **where pyannote-audio has been installed already** and run

```yaml
conda install -c coml shennong
```

Then you can pip-install the library in a regular way:

```
pip install pyannote.features.shennong
```

## Pyannote Configuration

To use features from this library, this is the kind of configuration you
should add your `config.yml` file:

```yaml
feature_extraction:
   name: pyannote.features.ShennongMfcc
   params:
       coefs: 19
       e: True
       D: True
       DD: True
       mfccWindowType: 'hanning'
       mfccLowFreq: 20
       mfccHighFreq: -100 # Real value will be (f_nyquist - 100)
       with_pitch: True
       with_cmvn: False
       duration: 0.025
       step: 0.010
       sample_rate: 16000
       pitchFmin: 20
       pitchFmax: 500
```