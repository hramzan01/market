# Market Project Plan

Building a data product is daunting 😱 but don’t worry, we’ve got you covered! 

# DAYS 1 & 2: 05, 07, and 09 March 
Discuss with your teams and write down answers to these questions:
What are we solving?

What are we predicting?

Why are we predicting? 

What are you demoing? How do you picture demo-day? 

# Get the Data 
List the data you need and how much you need.
Find and document where you can get that data.
Check how much space it will take.
Check legal obligations, and get authorisation, if necessary.
Create a workspace (with enough storage space).
Convert the data to a format you can easily manipulate.
Ensure sensitive information is deleted or protected (e.g., anonymized).
Check the size and type of data (time series, sample, geographical, etc.).
Sample a test set, put it aside for the final stage (no data snooping!).

Once your team has scoped out the problem space we can move into the exploratory and cleaning phases. 

# DAY 3 - 12 and 14 March 
Explore the Data
Create a copy of the data for exploration (sampling it down to a manageable size if necessary).
Use Jupyter notebooks to keep a record of your data exploration.
Study each attribute and its characteristics:
Name • Type (categorical, int/float, bounded/unbounded, text, structured, etc.)
% of missing values
Noisiness and type of noise (stochastic, outliers, rounding errors, etc.)
Possibly useful for the task?
Type of distribution (Gaussian, uniform, logarithmic, etc.)
For supervised learning tasks, identify the target attribute(s).
Visualise the data.
Study the correlations between attributes.
Study how you would solve the problem manually.
Identify the promising transformations you may want to apply.
Identify extra data that would be useful
Document what you have learned.

# DAY 4 - 16 March
Prepare the Data & Start Packaging
Work on copies of the data (keep the original dataset intact).
Data cleaning:
Check out the data preparation lecture
Write functions for all data transformations you apply, so you can easily prepare the data the next time you get a fresh dataset and you can apply these transformations in future projects:
To clean and prepare the test set
To clean and prepare new data instances once your solution is live
To make it easy to treat your preparation choices as hyperparameters

By the end of the day, you should have the bare bones of a package with a data pipeline.

# DAY 5 & 6 -  19, 21, 23 March
Shortlist, then refine models 

Hopefully you have built a quick and dirty pipeline, we want to identify as many pitfalls within the product as quickly as possible.
			
If the data is huge, you may want to sample smaller training sets so you can train many different models in a reasonable time (be aware that this penalises complex models such as large neural nets or Random Forests).
You will want to test all those models within your package rather than your notebook.
After training a basic model, train many quick and dirty models from different categories (e.g., linear, naive Bayes, SVM, Random Forests, neural net, etc.) using standard parameters.
Measure and compare their performance
Remember to use K-fold validation
Shortlist a couple of models at this stage
Models are an iterative process (at this point you’ll start making it better).

Transfer code to python scripts to prep package

At this stage your package should be looking much more like the end product, ensuring your data preparation and modelling scripts are ready to go into production.

# DAY 7 - 26 and 28 March 
Front end
At this point, you should think about how your product will interact with the real world. Discuss with your team how your application will look and work using streamlit. Make sure you have some kind of frontend structure that you can describe/show to the teaching team.
Try and capitalise on the capabilities of Streamlit and draw out the user flow, tying back interactions to backend pseudo code

# DAY 8 - 30 March 
Fine Tuning				
At this point, your team should be thinking about:
Supporting the front end interactions with back end code
Fine-tuning the shortlist of models to get the best performance
Remember you can squeeze out higher performance by looking at which mistakes your model makes and try to introduce new features.
Once confident with performance, evaluate the test set for generalisation
First Draft of Presentation
Now your team is confident with the application, start working on your presentation and smooth out any interactions within the app . Think about telling a story to the audience:
Highlighting the big picture
How you’ve solved the problem
Any interesting findings
Future work
Obstacles 

# DAY 9 - 02 and 04 April
Practice, Practice, Practice!
We will drill you on your presentation in each of these sessions so you are feeling prepared. You will have a small amount of time to tweak things, but no major changes or updates!!

# DAY 10 → 06 April is DEMO DAY!!!
