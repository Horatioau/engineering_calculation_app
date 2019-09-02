# Engineering Python
Something about what this does...
## Installation
You will need to have previously installed python3, see https://www.python.org/downloads/ for details.

To install the app to run locally, run the following from command-line:
```
git clone https://github.com/andrewthomasjones/engineering_calculation_app.git
cd engineering_calculation_app
python3 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:8050/ in your browser to view the app.
You can stop the running app using CRTL+C.

When you are finished you can deactivate the python virtual environment using:
```
deactivate
```

## To-Do list
* Enter ground values
* Export gantry and trestke specs as .csv
* Clean up GUI
* Add additional calculations
* Add unit-testing
* Set up CI pipeline
