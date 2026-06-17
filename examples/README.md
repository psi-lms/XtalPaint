# Examples of using the XtalPaint package


## Without using AiiDA

The `running-wo-AiiDA.ipynb` notebook shows how to use the XtalPaint package without AiiDA. It is a good starting point for users who want to get familiar with the package and its functionality. All the methods that are otherwise wrapped in the AiiDA WorkGraphs are just executed serially in this notebook.
The latest symmetry refinement and uniqueness analysis steps are not included. However, they are not specific to this package and can easily be integrated by the user if needed.

## Using AiiDA

The `running-with-AiiDA.ipynb` notebook shows how to use the XtalPaint package with AiiDA. The advantage of using AiiDA is that it keeps track of the data and also enables remote submission to different HPC clusters from your local machine. Moreover, WorkGraphs are already predefined with several optional steps.
