# People Flows with TimeSformer" 

## Dataset

&emsp;1. Download FDST Dataset from
Google Drive: [link](https://drive.google.com/drive/folders/19c2X529VTNjl3YL1EYweBg60G70G2D-w) 

&emsp;2. Create the hdf5 files with make_dataset.py, you need to set the path according to dataset location.

&emsp;3. Use create_json.py to generate the json file which contains the path to the images.

## Training
In command line:

```
python train.py train.json val.json

``` 

The json files here are generated from previous step (Dataset. 3.)

## Testing
&emsp;1. Modify the "test.py", make sure the path is correct.

&emsp;2. In command line:

```
python test.py

``` 

## Visualization
&emsp;1. Modify the "plot.py", make sure the path is correct.

&emsp;2. In command line:

```
python plot.py

``` 
This will plot the flows of each direction along with the density map

## Visualization of Attention Map
&emsp;1. Modify the "SA_explain_all.py", make sure the path is correct.

&emsp;2. In command line:

```
python SA_explain2.py

``` 
This will plot the attention map of one frame for each scene.
The SA_explain.py and plot_attention_maps.py can be used to plot a single selected frame


## Pre-trained Model

The pretrained model is in [GoogleDrive](https://drive.google.com/file/d/1dMcUwrW0R2QEoY77dvUJxpt36_YR9hNM/view?usp=sharing)

