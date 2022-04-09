# common-models

### Getting started
##### Cloning the repo
First, clone the repo:

```
git clone https://github.com/emotive-computing/common-models.git
```

Also note that Python 3 is used in this project and the version of pip installed on your computer will need to be compatible with Python 3. Please install these versions of Python/pip if you don't have them. 

##### Optional: Create a virtualenv for pip dependencies
Then, you might want to create a virtualenv for pip dependencies. These steps pertain to how this might be done specifically on Windows. Steps will vary somewhat for other OS's.

First, the virtual environment can be created by using venv.

```
python3 -m venv virt_env
```


_Note: if you have issues getting the python3 command to work, and you have Python3 installed, you may need to add the folder to Python3 to your path, and rename the python.exe file in this folder to python3.exe. For me, the folder to Python3 I needed to add was C:\Users\cathlyn\AppData\Local\Programs\Python\Python36\Scripts. Of course, you can always continue to just use the python command to execute, if Python 2 is not also installed. This was just something I needed to do._

_Another Note: Tensorflow currently will not work with Python 3.7, so make sure you are using 3.6 or less. I currently am using Python 3.6.2 and haven't tested with other Python 3 versions, but hopefully they should still work. Earlier versions before 3.5 might also require the use of pyvenv rather than venv for creating virtual environments. Let me know if there are any issues with other versions of Python3._

Activate the virtualenv (at the same level as the project folder might be a good location, so that the commonapp and virt_env folders are next to each other). The command for this is:

```
Windows:
.\virt_env\Scripts\activate

Linux:
source ./virt_env/bin/activate
```

If successful, (virt_env) should show at the start of each line of the command prompt until the virtualenv is deactivated by typing running the deactivate command, found in the same folder as the activate.

##### Optional: Create a Conda Environment
This works well for Windows users. Do not create a virtual environment (previous section) if you choose to use Anaconda.

Download and install [Anaconda](https://conda.io/docs/user-guide/install/windows.html "Anaconda install"). Make sure to test your install like the website says.

Add the conda scripts command to your environment variables. If you installed for all users, this will be in C:\ProgramData\Miniconda3\Scripts. Restart your computer after adding the environment variable.

Create your environment. You can use any environment name you want. Here, it is "modeling."

```
conda create -n modeling python=3.6.2
```

Activate the environment you just created. After you do this, you should see the name of your environment at the beginning of the command prompt line.

```
activate modeling
```

Double check your python version. If it is wrong, consult the [Anaconda website](https://conda.io/docs/user-guide/tasks/manage-python.html "Python Versions").

```
python --version.
```

Now you can navigate to the main project folder.

##### Install required dependencies 

Make sure your working directory is the main project folder. Generally after cloning the repo you will need to go down one folder into the commonapp repo:

```
cd commonapp
```

Then install dependencies with the below command. Note that this project requires python 3, and pip rather than pip may need to be used here depending on how it is setup on your machine. In addition, make sure you have activated the virtual environment prior to running the install command should a virtual environment be desired.

```
pip install -r requirements.txt
```

Next install the package itself:

```
pip install .
```

Hopefully everything installed without any problems! If you did encounter any errors, see the next section on troubleshooting for some tips.


##### Nltk downloads
You will need to make sure necessary nltk data is downloaded. From the main project directory, type the command:

```
download-nltk
```

This should only need to be run once.

### Input data file format
First, a datafile will be needed for input to the program. The file should be in a csv file in the format of:

ID | Text | Label1 | Label2 | ...

* The first row should be the headings, and each row after that should give one data sample. 
* Please note the headings can be given any name, be in any order, and have any number of labels to predict. 
* Additional columns which are not to be used are ok to be in the file. 
* The ID is whatever the unique identifier for each data item is. The Text is the text to classify. Labels are the y-labels to predict.

### Configuration of settings file
A file can be created to configure the settings for running the dataset. See the settings-hcmw.py or settings-commonapp.py files as examples.

Details such as the tasks to be run, models to be used, parameters to be cross-validated, names of the columns corresponding the the ID, Text, and Labels, name of the data input file, name of the folder to output results, number of cross validation folds, and other settings can all be adjusted in this file.

See CONFIGURING_SETTINGS.md for more details.

##### Config folder and config files
In addition, there is a folder with configs for different things. However, in general these should not need to change unless you are adding functionality to the app. As far as just configuring different properties for different datasets / preferences, changes should be limited to the settings-whatever.py file.


##### GloVe file(s) for pretrained word embeddings
Based on how the settings file is configured, pretrained word vectors may be needed. 

I tried to go ahead and include the pretrained gloVe file for use, but it was too large to check into the repo. So you will need to download it yourself and include it, if the setting to use pretrained gloVe embeddings is selected. This file can be found on [gloVe's website](https://nlp.stanford.edu/projects/glove "Download pretrained vectors"). 

From the zipped folder that is downloaded, copy and paste the file glove.6B.100d.txt into a folder called include/ in the main project repo. Alternatively you can put the file in another location and just change the path to this file in the config/embedding_config.py file. Either way should work.


So far, I've just used the file with 100-dim word vectors from Wikipedia, but of course different files could have been used, and a value in the embedding config file can be set to the name of this vectors file so that the app will pull the right vectors file to use. The EMBEDDING_DIM constant would just need to be set to the dimension of the vectors.

In addition to using the files provided by gloVe, custom word vector files can be found on a given set of test data using the gloVe algorithm. I'll try to find some time to add the full details of how to do this here, but the directions on gloVe's website (see the link above) help to give a general idea.


### Running the program
Run the main command passing the name of the settings file as input:

```
main --settings settings-whatever.py
```

The specific programs that will be run and output that will be generated depend on the tasks that were configured to be run in the settings file. Results will be output into the file specified in the settings file.

### Generating combined results or word clouds after running 
After running the main program, you can use the same settings file to generate word clouds if you did not do so previously using the command:

```
word-clouds --settings settings-whatever.py 
```

You can also generate csv files to combine all results from all models and all predictions from all models using the command:

```
analyze-results --settings settings-whatever.py
```

### Running code on AWS
For info, see RUNNING_ON_AWS.md.

### Examples
See the examples folder for how to run for more specific use cases, such as for classification vs. regression and for language modeling vs. just using features.

### Questions or Issues
If your run into any problems or have any questions, please email me at cathlyn.stone@colorado.edu.

