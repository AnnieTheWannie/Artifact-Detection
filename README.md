# Artifact Detection
For this project a method is designed to detect artifacts in PICU monitoring data. More specifically in mean blood pressure due to blood sampling and artifact dips in heart rate and oxygen saturation. 

In DipCharactization.py containts the class where the dips satisfying some artifact characteristics are identified and produces a dataframe with time stamps of the beginning and end of an artifact.
In MakeIBPDataframe
