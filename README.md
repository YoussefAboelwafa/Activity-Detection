# Activity Detection

## Dataset
Daily & Sports Activity [Dataset](https://www.kaggle.com/datasets/obirgul/daily-and-sports-activities/data)

#### Brief Description of the Dataset:

Each of the 19 activities is performed by eight subjects (4 female, 4 male, between the ages 20 and 30) for 5 minutes. <br>
Total signal duration is 5 minutes for each activity of each subject.<br>
The 5-min signals are divided into 5-sec segments so that 480(=60x8) signal segments are obtained for each activity. <br>

#### The 19 activities are:
- sitting (A1)
- standing (A2)
- lying on back and on right side (A3 and A4)
- ascending and descending stairs (A5 and A6)
- standing in an elevator still (A7)
- and moving around in an elevator (A8)
- walking in a parking lot (A9)
- walking on a treadmill with a speed of 4 km/h (in flat and 15 deg inclined positions) (A10 and A11)
- running on a treadmill with a speed of 8 km/h (A12)
- exercising on a stepper (A13)
- exercising on a cross trainer (A14)
- cycling on an exercise bike in horizontal and vertical positions (A15 and A16)
- rowing (A17)
- jumping (A18)
- and playing basketball (A19)

#### File structure:

- 19 activities (a) (in the order given above)
- 8 subjects (p)
- 60 segments (s)
- 5 units on torso (T), right arm (RA), left arm (LA), right leg (RL), left leg (LL)
- 9 sensors on each unit (x,y,z accelerometers, x,y,z gyroscopes, x,y,z magnetometers)

Folders a01, a02, …, a19 contain data recorded from the 19 activities. <br>

For each activity, the subfolders p1, p2, …, p8 contain data from each of the 8 subjects. <br>

In each subfolder, there are 60 text files s01, s02, …, s60, one for each segment.
