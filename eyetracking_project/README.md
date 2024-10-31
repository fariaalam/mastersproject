# Application for Editing Articles based on Visual Attention of Reviewers
In this project we present an innovative approach to enhancing the process of reviewing and editing scientific texts through the use of eye-tracking. By focusing on the area of interest (AOI) at the paragraph level, our study explores the potential of eye gaze data to provide detailed feedback on how readers engage with text documents( such as read thoroughly, skimmed or skipped). In contrast to previous researches that concentrated on document-level analysis, our work tries to detect reading patterns of individual and collective readers at the AOI level, offering more precise insights into reader engagement. The software developed for this project converts PDF pages to images for display and collects eye-tracking data from users with the help of Tobii Eye-Tracker. Then the collected gaze points are normalized to ensure consistency across different screen sizes and resolutions, making the data device-independent. This normalization is key to accurately interpreting which parts of the document capture the reader’s attention. We have used several eye-tracking matrices to detect reading patterns. Overall, the project presents a novel method for writers to receive feedback on their work, potentially improving the readability and engagement of their texts based on real user interaction data in the form of eye gaze data.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

Python >= 3.8 and a package manager like pip should be installed on your machine.
OS: Linux, Mac, Windows


### Installing

Clone the repository to your local machine. Navigate to the App directory and install the required packages using pip:

Install the requirements from **requirements.txt** which is in **code/appfile** using pip : 
```
pip install -r requirements.txt 
```
We have used pytessaract for OCR purpose to install it properly go to this link [Homebrew](https://brew.sh/)
and install Homebrew (we all have used Mac, not sure about linux and windows ). Then run this command in terminal to install pytessaract properly: 
```
brew install tesseract 
```

## Running the Application

Copy the pdf file to **data/pdfs folder**, on which you want to conduct the experiments.  
Edit the  **config.ini**  file in **code/App** folder 
Put the pdf file's name in the **pdf_path** variable
To run the application, navigate to the App directory and run the **main.py** file:

```
python main.py
```

When the application is running you will see 4 buttons if you want to collect gaze data click 
the **reader** button. In the new screen you will see the first page of the pdf you have mentioned
in the right side you will see the **start recording** button. Click it start recording data. The log 
file will be saved in the **data/pdfs** folder in folder named with the structure like this
**log_files_pdfname_date_time**. Log file of all the pages will be saved in that folder.

If want to do analysis click on **analysis** button. In the new screen you can see analysis of the 
individual users and also all of the users combined. You can also see reading patterns of readers.
There are different buttons, just play with all of them. When the **combined** toggle button is 
clicked and you also click the **read thoroughly**, **skimmed** or **skipped** button then you can 
click on any paragraphs and see paragraph related analysis in the **analysis report** part of the 
screen.

## Project Structure

The project is structured as follows:

- `code/App/eyeTracker`: Contains the eye tracker classes usable in the app.
- `code/App/widgets/`: Contains the widget classes for the app.
- `code/App/screens/`: Contains the screen classes for the app.
- `code/App/config.ini`: Contains the settings for the app.
- `code/App/resources/kv`: Contains the kv files of the app.
- `code/App/utils`: Contains the helper functions of the analysis screen.
- `data/pdfs`: Contains pdfs and logfiles.

## Built With

- [Python](https://www.python.org/)
- [Kivy](https://kivy.org/#home)

## Authors
Faria Alam (faal00001@stud.uni-saarland.de)
Andreas Koch (anko00009@uni-saarland.de)
Oguzhan Sanlitürk (ogsa00001@stud.uni-saarland.de)

## Project Status
Completed