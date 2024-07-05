# Artifact Detection
For this project a method is designed to detect artifacts in PICU monitoring data. More specifically in mean blood pressure due to blood sampling and artifact dips in heart rate and oxygen saturation. 

- In DipCharactization containts the class where the dips satisfying some artifact characteristics are identified and produces a dataframe with time stamps of the beginning and end of an artifact.
- ChangePointDetector is mainly used for the blood sample detection in MakeIBPDataframe. But also provides functions to detect other changes, like sensor dysfunction. 
- In MakeDataframeIBP a dataframe is made of all the blood sample event characteristics.
