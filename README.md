# Photo Editor
This project is a simple yet powerful photo editor build with Python. The main goal is to offer users a straightforward interface for performing basic image editing tasks with the ability to undo actions and view the history of changes.

## Features
- ***Adjust Brightness and Contrast*** - Easily adjust the brightness and contrast levels of your image.
- ***Convert to Grayscale*** - Transform color images into grayscale.
- ***Add Padding*** - Add custom padding (border) around the image with various settings (constant, reflect, replicate borders, square, rectangle and custom ratio).
- ***Apply Thresholding*** - Convert image to black & white using binary on inverse thresholding technique.
- ***Blend with Another Image*** - Blend two images together with adjustable alpha transparency.
- ***Undo Last Operation*** - Cancel the last change and go back to the previous image state.
- ***History Tracking*** - See the log of all changes applied to the image.
- ***Save the Result*** - Export the edited image.

## How to Run & Use the program
1. Clone the repo using `git clone` command and navigate to the project directory `cd photo-editor`.
2. Install the required libraries `pip install -r requirements.txt`.
   <br>Make sure that Python is installed (version 3+).
4. Move your image(s) you would like to edit to it to the project directory.
   <br>You can remove tester images (`image.jpg` and `overlay_image.jpg`).
5. Start the program by running `python main.py`.
6. Use the provided menu to select the desired operation and follow the on-screen instructions.
7. To save your changes and exit, select option `9` from the program menu.

## Tools used
Photo Editor is solely implemented using Python and is build using these libraries:
- OpenCV - for image processing and editing;
- Matplotlib - the heart of the graphical user interface in this program;
- NumPy - for numerical operations for image editing.

## Check out some examples

**Contrast Adjustment**
<br>
<img width="700" alt="contrast adjustment example" src="https://github.com/user-attachments/assets/cf0d0a25-0385-4ad4-b659-fa7b443fccee" />
<br>

**Image Blending**
<br>
<img width="700" alt="image blending example" src="https://github.com/user-attachments/assets/ab4c30a6-2bed-488c-b9fe-eb6837367d5d" />
<br>

**Thresholding**
<br>
<img width="700" alt="thresholding example" src="https://github.com/user-attachments/assets/20c4acac-0188-4fc3-b69d-997700ef0808" />
<br>
