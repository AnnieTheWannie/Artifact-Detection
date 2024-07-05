# Artifact Detection
This is part of my master thesis for applied data science. For this project a method is designed to detect artifacts in PICU monitoring data and also describe them. More specifically in mean blood pressure due to blood sampling and artifact dips in heart rate and oxygen saturation.

- In DipCharactization containts the class where the dips satisfying some artifact characteristics are identified and produces a dataframe with time stamps of the beginning and end of an artifact.
- SensorDysfunction detects periods that are almost certain due to sensor dysfunction and have no physiological cause.
- ChangePointDetector contains functions for smoothing. 
- In MakeDataframeIBP a dataframe is made of all the blood sample event characteristics.
