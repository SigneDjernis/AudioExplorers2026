# AudioExplorers2026
Welcome to the **AudioExplorers2026** repository! This project is part of the Oticon Audio Explorers 2026 Software Engineering challenge, where we develop a complete audio processing pipeline to localize, separate, enhance and analyse multiple speakers in complex auditory scenes, also know as the Cocktail Party Problem.

## Overview
In this repository, we explore a set of complementary methods compiled into a processing pipeline. First, the direction of arrival (DoA) of each speaker is estimated using a Steered Response Power (SRP) algorithm. Next, beamforming techniques are applied to separate the individual speakers based on their spatial locations. Finally, the processed audio is analysed to extract relevant information about the auditory scene, including language identification, speaker characteristics (such as gender) and content of the conversation through transcription.

## Features
- **Steered Response Power (SRP) algorithm:** To detect the direction of arrival (DoA) of each speaker, we have implemented SRP ('steered_response_power.py'), which returns the angles of interest.
- **Beamforming algorithm:** Given the estimated angles of interest, beamforming techniques are applied to separate individual speakers based on their spatial locations ('beamforming.py')
- **Characteristics algorithm:** Using the separated audio signals, this module ('characteristics.py') transcribes the speech and classifies speaker characteristics such as gender.


## Structure
The repository is organized as follows:
- recordings: Contains the 'example_mixture.wav' and 'wav'
- funcs: contains the scripts from each of the three analysis sections.
- speakers: Contains the separated audio signals for all the angles of interest
- results: contains images from the different and the transcriptions.
- Outer folder: Contains code for solving the Software Challenge.


## Requirements
This project uses Python as its main programming language. Ensure you have Python installed along with the necessary dependencies.

Run the following to install the required libraries:
```bash
pip install -r requirements.txt
