This repo was designed to train most accurate and at the same time fastest face landmarks and pose from single photo estiamtion algorithm

**It contains:**

 - facilities to prepare training data from video and pictures (pictures plus json annotations), annotations provided by SOTA open source [landmarks](https://github.com/1adrianb/face-alignment) and [pose](https://github.com/natanielruiz/deep-head-pose) detectors

 - facilities to train [Dlib](http://dlib.net) models

**Typical pipeline:**

 1. Grab videos or photos from somwhere

 2. Annotate frames/photos by provided [tools](/annotataion_util/annoface.py) 

 3. Crop faces by [cropface](https://github.com/pi-null-mezon/OpenFRT) utility

 4. Learn regressors by provided learners
